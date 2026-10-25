"""Native inference function and permanent child-only accidental-spawn guard.

One inherited socket carries a bounded immutable execution and chunked result.
Calling infer alone does not establish process or terminal authority.
"""
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.transport import Refused

_guard_installed = False


class SpawnRefused(RuntimeError):
    pass


def _refuse_spawn(*args, **kwargs):
    raise SpawnRefused('native_child_spawn_forbidden')


def _audit_spawn(event, args):
    if event in {'subprocess.Popen', 'os.fork', 'os.forkpty', 'os.posix_spawn',
                 'os.system', 'os.exec', 'os.spawn', 'pty.spawn'}:
        _refuse_spawn()


def install_spawn_guard():
    """Permanent, process-global: call only on entry to the owned exec child.

    Audit plus explicit hooks cover ordinary Python process creation, including
    multiprocessing/resource_tracker. This is not a native-code sandbox.
    """
    global _guard_installed
    if _guard_installed:
        return
    sys.addaudithook(_audit_spawn)
    import subprocess
    from multiprocessing.process import BaseProcess
    subprocess.Popen = _refuse_spawn
    BaseProcess.start = _refuse_spawn
    for name in ('fork', 'forkpty', 'posix_spawn', 'posix_spawnp', 'system',
                 'execl', 'execle', 'execlp', 'execlpe', 'execv', 'execve',
                 'execvp', 'execvpe', 'spawnl', 'spawnle', 'spawnlp', 'spawnlpe',
                 'spawnv', 'spawnve', 'spawnvp', 'spawnvpe'):
        if hasattr(os, name):
            setattr(os, name, _refuse_spawn)
    _guard_installed = True


def infer(execution, text, bundles, references, inventory, *,
          reference_temp_dir=None, managed_owner_uid=None):
    """Real input verifier/complete loader/encoder with a closed native backend.

    The owned entry installs spawn_guard before calling this function. No
    arbitrary backend factory or caller-selected method crosses this boundary.
    """
    from worker.engines.qwen_execution import current_native_hardware
    from worker.engines.qwen import verify_service_inputs, execute
    if not _guard_installed:
        raise Refused('native_spawn_guard_required')
    current_native_hardware(execution, inventory)
    if execution.timeout_s > 30 or execution.max_output_bytes > 1048576:
        raise Refused('native_execution_limit')
    _, raw = verify_service_inputs(execution, bundles, references, inventory,
                          owner_uid=managed_owner_uid)
    from worker.engines.qwen_prepare import QwenPreparation, selected_reference
    if type(execution) is QwenPreparation: selected_reference(execution, raw)
    from lib.tts_backends.qwen_native import NativeQwenBackend
    return execute(execution, text, bundles, references, inventory,
        reference_temp_dir=reference_temp_dir, managed_owner_uid=managed_owner_uid,
        backend_factory=NativeQwenBackend)


def wire_send(connection,value,cutoff):
    """Private inherited socket, same bounded canonical framing, no TLS fiction."""
    import struct
    from worker.transport import canonical, remaining
    from worker.native_protocol import FRAME_LIMIT
    raw=canonical(value)
    if not 0<len(raw)<=FRAME_LIMIT: raise Refused('native_child_frame')
    remaining(connection,cutoff); connection.sendall(struct.pack('!I',len(raw))+raw)


def wire_receive(connection,cutoff):
    import json
    import struct
    from worker.transport import exact, canonical
    from worker.native_protocol import FRAME_LIMIT
    size,=struct.unpack('!I',exact(connection,4,cutoff))
    if not 0<size<=FRAME_LIMIT: raise Refused('native_child_frame')
    def pairs(items):
        value={}
        for k,v in items:
            if k in value: raise Refused('duplicate_key')
            value[k]=v
        return value
    def nonfinite(_): raise Refused('nonfinite')
    raw=exact(connection,size,cutoff)
    try:
        value=json.loads(raw,object_pairs_hook=pairs,parse_constant=nonfinite)
        if canonical(value)!=raw: raise Refused('native_canonical')
        return value
    except (ValueError,TypeError,UnicodeError,RecursionError): raise Refused('native_child_frame') from None



def proposal(snapshot,result,spec):
    import hashlib
    import secrets
    from worker.native_protocol import generated_source
    job,context,assets,profiles=snapshot; attempt=job['attempts'][-1]
    asset=dict(id='asset-'+secrets.token_hex(16),sha256=result['sha256'],frames=result['frames'],sample_rate=spec.sample_rate,channels=1,media_format='wav',media_subtype='PCM_16',provenance=dict(kind='generated',source_sha256=result['sha256'],source_id=generated_source(context),missing_fields=[]))
    message=dict(protocol_version='1.3.0',kind='result',job_id=job['id'],attempt_id=attempt['id'],generation=job['generation'],input_sha256=job['input_sha256'],input_asset_ids=assets,staging_id=attempt['staging_id'],owner_id=attempt['owner_id'],identity=context['plan']['resolved']['identity'],deadline=job['deadline'],stage='completed',result_asset_ids=[asset['id']],plan=context['plan'])
    manifest={k:message[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')}
    if context['plan']['operation'] == 'voice_prepare':
        from worker.engines.qwen_prepare import output_assets
        from worker.synthetic import candidates
        outputs = output_assets(context, result)
        proposed = candidates(message, outputs[1], outputs[0])
        message['result_asset_ids'] = [a['id'] for a in outputs]
    else: outputs = [asset]; proposed = []
    manifest.update(operation=context['plan']['operation'],artifact_id='artifact-'+secrets.token_hex(16),outputs=outputs,candidates=proposed)
    return dict(message=message,manifest=manifest)


def validate_proposal(snapshot,value,data,spec):
    import hashlib
    from contracts import validate as oracle
    from worker.native_protocol import _closed
    from worker.native_supervisor import pcm
    from worker.native_protocol import generated_source
    _closed(value,('message','manifest'))
    job,context,assets,profiles=snapshot
    oracle.worker_message(value['message'],job,assets,transport_owner=job['attempts'][-1]['owner_id'],context=context,current_profiles=profiles)
    oracle.validate('result_manifest',value['manifest'])
    m=value['manifest']; message=value['message']
    if (message['kind'] != 'result' or m['operation'] != context['plan']['operation'] or
            any(m[k] != message[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256'))):
        raise Refused('native_proposal')
    if context['plan']['operation'] == 'voice_prepare':
        from worker.engines.qwen_prepare import output_assets
        from worker.synthetic import candidates
        audition = dict(sha256=hashlib.sha256(data).hexdigest(),frames=(len(data)-44)//2)
        outputs = output_assets(context, audition)
        expected = candidates(message, outputs[1], outputs[0])
        from worker.transport import canonical
        if (canonical(m['outputs']) != canonical(outputs) or canonical(m['candidates']) != canonical(expected) or
                message['result_asset_ids'] != [a['id'] for a in outputs]): raise Refused('native_proposal')
        if len(data)>min(spec.max_output_bytes,1048576): raise Refused('native_output')
        pcm(data,spec)
        if len(data) != 44 + outputs[1]['frames']*2: raise Refused('native_output')
        return
    if message['kind']!='result' or any(m[k]!=message[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')) or len(m['outputs'])!=1 or m['candidates'] or m['operation']!='speech_generate': raise Refused('native_proposal')
    a=m['outputs'][0]
    if len(data)>min(spec.max_output_bytes,1048576) or a['sha256']!=hashlib.sha256(data).hexdigest() or message['result_asset_ids']!=[a['id']] or a['sample_rate']!=spec.sample_rate or a['provenance']!=dict(kind='generated',source_sha256=a['sha256'],source_id=generated_source(context),missing_fields=[]): raise Refused('native_output')
    pcm(data,spec)
    if len(data)!=44+a['frames']*2: raise Refused('native_output')


def run_socket(connection, *, inference=None, guard=True):
    """One typed immutable input, one bounded result, then EOF. Test seam is ML only."""
    import base64
    import hashlib
    import time
    from worker import native_protocol as p
    from worker.engines.qwen_execution import QwenExecution, inventory_from_wire
    config=wire_receive(connection,time.monotonic()+2)
    p._closed(config,('token','binding','snapshot','execution','bundles','references','inventory','conditioning','cutoff'))
    p._hex(config['token'],32); p._hex(config['binding'],64)
    if hashlib.sha256(p.canonical(config['snapshot'])).hexdigest()!=config['binding']: raise Refused('native_binding')
    cutoff=config['cutoff']
    if type(cutoff) not in (int,float) or not 0<cutoff-time.monotonic()<=30: raise Refused('native_deadline')
    if config['execution'].get('phase') == 'prepare':
        from worker.engines.qwen_prepare import QwenPreparation
        spec = QwenPreparation.from_wire(config['execution'])
    else: spec=QwenExecution.from_wire(config['execution'])
    if p.native_execution(config['snapshot'][1]) != spec: raise Refused('native_execution')
    from contracts import validate as oracle
    oracle.stored_context(config['snapshot'][0],config['snapshot'][1])
    if (spec.device,spec.dtype,spec.attention) != ('mps','float32','sdpa') or spec.task not in ('clone','custom_voice'): raise Refused('native_execution')
    if guard: install_spawn_guard()
    result=(infer if inference is None else inference)(spec,(config['snapshot'][1]['plan']['request']['audition_text'] if config['snapshot'][1]['plan']['operation']=='voice_prepare' else config['snapshot'][1]['plan']['request']['text']),config['bundles'],config['references'],inventory_from_wire(config['inventory']),reference_temp_dir=config['conditioning'],managed_owner_uid=os.getuid())
    value=proposal(config['snapshot'],result,spec)
    validate_proposal(config['snapshot'],value,result['data'],spec)
    wire_send(connection,dict(token=config['token'],binding=config['binding'],proposal=value,bytes=len(result['data']),sha256=result['sha256']),cutoff)
    for offset in range(0,len(result['data']),32768):
        wire_send(connection,dict(offset=offset,data=base64.b64encode(result['data'][offset:offset+32768]).decode()),cutoff)


def parent_exchange(connection,config,cutoff):
    import hashlib
    from worker import native_protocol as p
    wire_send(connection,config,cutoff)
    result=wire_receive(connection,cutoff)
    p._closed(result,('token','binding','proposal','bytes','sha256'))
    if (result['token'],result['binding'])!=(config['token'],config['binding']): raise Refused('native_token')
    p._int(result['bytes'],46,min(config['execution']['max_output_bytes'],1048576)); p._hex(result['sha256'],64)
    data=bytearray()
    while len(data)<result['bytes']:
        block=wire_receive(connection,cutoff); p._closed(block,('offset','data'))
        if type(block['offset']) is not int or block['offset']!=len(data): raise Refused('native_offset')
        data.extend(p.chunk(block['data']))
        if len(data)>result['bytes']: raise Refused('native_output')
    from worker.transport import remaining
    remaining(connection,cutoff)
    if connection.recv(1)!=b'' or hashlib.sha256(data).hexdigest()!=result['sha256']: raise Refused('native_output')
    return result['proposal'],bytes(data)


def main():
    import socket
    if len(sys.argv)!=2 or not sys.argv[1].isdigit(): raise Refused('native_entry')
    connection=socket.socket(fileno=int(sys.argv[1]))
    try: run_socket(connection)
    finally: connection.close()


if __name__=='__main__':
    main()
