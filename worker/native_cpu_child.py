"""One guarded native CPU preset job on an inherited private socket."""
import os
import sys
if __name__=='__main__':
    sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.transport import Refused
from worker.native_child import wire_send, wire_receive, install_spawn_guard
from worker import native_cpu_protocol as p
from worker.engines.native_kokoro import NativeKokoroExecution, STARTUP_SECONDS, REQUEST_SECONDS, JOB_SECONDS

CHILD_VERSION='native-kokoro-cpu-child-v1'


def validate_cutoff(cutoff,now,limit):
    import math
    if type(cutoff) not in (int,float) or not math.isfinite(cutoff) or not 0<cutoff-now<=limit:
        raise Refused('native_cpu_deadline')


def infer(context,bundles,*,on_loaded):
    """Called only after the permanent guard; no public filesystem authority."""
    from worker import native_child
    if not native_child._guard_installed:raise Refused('native_spawn_guard_required')
    from service.execution_context import verify_local
    from lib.tts_backends.kokoro_native import validate_host, validate_versions
    validate_host();validate_versions()
    spec=p.native_execution(context)
    verify_local(context,bundles)
    from worker.engines.kokoro import execute
    from lib.tts_backends.kokoro import KokoroBackend
    from lib.spoken_text import speech_text
    import inspect
    import numpy as np
    class NativeBackend(KokoroBackend):
        def synthesise(self,*args,**kwargs):
            result=super().synthesise(*args,**kwargs)
            if np.asarray(result.samples).dtype!=np.dtype('float32'):raise Refused('native_cpu_dtype')
            return result
    def factory(root,**kwargs):
        backend=NativeBackend(root,**kwargs)
        try:
            if backend._providers!=('CPUExecutionProvider',) or spec.preset not in backend._voices:
                raise Refused('native_cpu_backend')
            if inspect.signature(backend.model.create).parameters['speed'].default!=1.0:
                raise Refused('native_cpu_speed')
            floating=[i.type for i in backend.model.sess.get_inputs() if i.type.startswith('tensor(float')]
            if not floating or any(t!='tensor(float)' for t in floating):raise Refused('native_cpu_dtype')
            on_loaded()
        except BaseException:
            backend.close();raise
        return backend
    return execute(spec.model,speech_text(context['plan']),bundles,backend_factory=factory)


def _error(exc,stage):
    return dict(code=getattr(exc,'code','native_cpu_'+stage+'_failed'),type=type(exc).__name__,
        message=str(exc)[:1000],cause_type=type(exc.__cause__).__name__ if exc.__cause__ else None,
        cause=str(exc.__cause__)[:1000] if exc.__cause__ else None)


def run_socket(connection,*,inference=None,guard=True):
    import base64
    import time
    from worker.native_child import proposal,validate_proposal
    if guard:install_spawn_guard()
    config=wire_receive(connection,time.monotonic()+2)
    p._closed(config,('version','token','binding','snapshot','execution','bundles','inventory','cutoff'))
    if config['version']!=CHILD_VERSION:raise Refused('native_cpu_child_version')
    p._hex(config['token'],32);p._hex(config['binding'],64)
    import hashlib
    from worker.transport import canonical
    if hashlib.sha256(canonical(config['snapshot'])).hexdigest()!=config['binding']:
        raise Refused('native_cpu_binding')
    cutoff=config['cutoff'];validate_cutoff(cutoff,time.monotonic(),JOB_SECONDS)
    startup=min(cutoff,time.monotonic()+STARTUP_SECONDS)
    spec=NativeKokoroExecution.from_wire(config['execution'])
    if p.native_execution(config['snapshot'][1])!=spec:raise Refused('native_cpu_execution')
    p.native_binding(config['snapshot'][1]);p.inventory(config['inventory'])
    from contracts import validate as oracle
    oracle.stored_context(config['snapshot'][0],config['snapshot'][1])
    stage='startup'
    try:
        def loaded():
            nonlocal cutoff,stage
            wire_send(connection,dict(kind='cpu-loaded-v1',token=config['token'],binding=config['binding']),startup)
            ack=wire_receive(connection,startup)
            p._closed(ack,('kind','token','binding','cutoff'))
            if (ack['kind'],ack['token'],ack['binding'])!=('cpu-run-v1',config['token'],config['binding']):raise Refused('native_cpu_token')
            candidate=ack['cutoff'];validate_cutoff(candidate,time.monotonic(),REQUEST_SECONDS)
            if candidate>config['cutoff']:raise Refused('native_cpu_deadline')
            cutoff=candidate;stage='generation'
        result=(infer if inference is None else inference)(config['snapshot'][1],config['bundles'],on_loaded=loaded)
        value=proposal(config['snapshot'],result,spec,protocol=p)
        validate_proposal(config['snapshot'],value,result['data'],spec,protocol=p)
        wire_send(connection,dict(token=config['token'],binding=config['binding'],proposal=value,
            bytes=len(result['data']),sha256=result['sha256']),cutoff)
        for offset in range(0,len(result['data']),p.CHUNK_LIMIT):
            wire_send(connection,dict(offset=offset,data=base64.b64encode(result['data'][offset:offset+p.CHUNK_LIMIT]).decode()),cutoff)
    except BaseException as exc:
        try:wire_send(connection,dict(kind='cpu-error-v1',token=config['token'],binding=config['binding'],error=_error(exc,stage)),min(startup if stage=='startup' else cutoff,time.monotonic()+1))
        except BaseException:pass
        raise


class ChildFailure(RuntimeError):
    def __init__(self,error):
        self.error=error
        super().__init__(error['code']+': '+error['message'])


def _receive(connection,cutoff,config):
    value=wire_receive(connection,cutoff)
    if type(value) is dict and value.get('kind')=='cpu-error-v1':
        p._closed(value,('kind','token','binding','error'))
        if (value['token'],value['binding'])!=(config['token'],config['binding']):raise Refused('native_cpu_token')
        error=value['error'];p._closed(error,('code','type','message','cause_type','cause'))
        for key in ('code','type','message'):
            if type(error[key]) is not str or len(error[key])>1000:raise Refused('native_cpu_error')
        for key in ('cause_type','cause'):
            if error[key] is not None and (type(error[key]) is not str or len(error[key])>1000):raise Refused('native_cpu_error')
        raise ChildFailure(error)
    return value


def parent_exchange(connection,config,cutoff,*,on_loaded):
    import hashlib
    import time
    if config['references'] or config['conditioning'] is not None:raise Refused('native_cpu_preset_only')
    wire={k:v for k,v in config.items() if k not in ('references','conditioning')};wire['version']=CHILD_VERSION
    spec=NativeKokoroExecution.from_wire(config['execution'])
    wire_send(connection,wire,cutoff)
    loaded=_receive(connection,cutoff,config);p._closed(loaded,('kind','token','binding'))
    if (loaded['kind'],loaded['token'],loaded['binding'])!=('cpu-loaded-v1',config['token'],config['binding']):raise Refused('native_cpu_token')
    request_cutoff=min(config['cutoff'],time.monotonic()+spec.timeout_s)
    validate_cutoff(request_cutoff,time.monotonic(),REQUEST_SECONDS);on_loaded(request_cutoff)
    wire_send(connection,dict(kind='cpu-run-v1',token=config['token'],binding=config['binding'],cutoff=request_cutoff),cutoff)
    cutoff=request_cutoff
    result=_receive(connection,cutoff,config);p._closed(result,('token','binding','proposal','bytes','sha256'))
    if (result['token'],result['binding'])!=(config['token'],config['binding']):raise Refused('native_cpu_token')
    p._int(result['bytes'],46,min(spec.max_output_bytes,1048576));p._hex(result['sha256'],64)
    data=bytearray()
    while len(data)<result['bytes']:
        block=_receive(connection,cutoff,config);p._closed(block,('offset','data'))
        if type(block['offset']) is not int or block['offset']!=len(data):raise Refused('native_cpu_offset')
        data.extend(p.chunk(block['data']))
        if len(data)>result['bytes']:raise Refused('native_cpu_output')
    from worker.transport import remaining
    remaining(connection,cutoff)
    if connection.recv(1)!=b'' or hashlib.sha256(data).hexdigest()!=result['sha256']:raise Refused('native_cpu_output')
    return result['proposal'],bytes(data)


def main():
    import socket
    if len(sys.argv)!=2 or not sys.argv[1].isdigit():raise Refused('native_cpu_entry')
    connection=socket.socket(fileno=int(sys.argv[1]))
    try:run_socket(connection)
    finally:connection.close()


if __name__=='__main__':main()
