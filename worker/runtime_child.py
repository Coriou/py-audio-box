"""Closed synthetic/Kokoro/Pocket/Qwen child. No management capability or service imports."""
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import socket
import sys
import time
from worker.transport import receive
from worker.isolated_synthetic import run


def run_generation(connection, config, staging, cutoff, *, backend_factory=None):
    """Authenticated one-shot execution; injected factory is an in-process test seam."""
    import hashlib
    from datetime import datetime, timezone
    from uuid import uuid4
    from contracts import validate as oracle
    from worker.transport import Session, Refused, authenticate, canonical, receive, send
    from worker.runtime_protocol import execution, generated_source
    from worker.engines.kokoro import execute
    job, context = config['job'], config['context']
    spec = execution(context)
    if spec is None: raise Refused('execution_required')
    attempt = job['attempts'][-1]
    cutoff = min(cutoff, time.monotonic()+max(0, (datetime.fromisoformat(job['deadline'])-datetime.now(timezone.utc)).total_seconds()))
    binding = hashlib.sha256(canonical([job, context, config['allowed_assets'], config['profiles']])).hexdigest()
    authenticate(connection, Session(config['secret'], binding, config['control_uid'], config['gid']), cutoff, control=False)
    message = receive(connection, cutoff)
    oracle.worker_message(message, job, config['allowed_assets'], transport_owner=attempt['owner_id'], context=context, current_profiles=config['profiles'])
    if message['kind'] != 'execute': raise Refused('unexpected_message')
    if spec.engine == 'qwen':
        from worker.engines.qwen import execute
        from worker.engines.qwen_execution import inventory_from_wire, current_hardware
        inventory = inventory_from_wire(context['execution']['hardware'])
        current_hardware(spec,inventory)
        stage = attempt['staging_id']
        if Path(stage).name != stage or stage in ('.', '..'): raise Refused('invalid_stage')
        if spec.task == 'clone':
            directory = os.open(Path(staging)/stage, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try: os.mkdir('conditioning',0o700,dir_fd=directory)
            finally: os.close(directory)
        result = execute(spec,context['plan']['request']['text'],context['execution']['bundles'],
                         context['execution']['references'],inventory,
                         reference_temp_dir=Path(staging)/stage/'conditioning' if spec.task == 'clone' else None,
                         backend_factory=backend_factory,managed_owner_uid=10001)
    elif spec.engine == 'pocket-tts':
        from worker.engines.pocket import execute
        stage = attempt['staging_id']
        if Path(stage).name != stage or stage in ('.', '..'): raise Refused('invalid_stage')
        # Attempt-owned; never remove while descendants or death proof are unknown.
        directory = os.open(Path(staging)/stage, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            os.mkdir('conditioning', 0o700, dir_fd=directory)
        finally:
            os.close(directory)
        preparing = context['plan']['operation'] == 'voice_prepare'
        if preparing:
            from worker.engines.pocket_prepare import execute
        text = context['plan']['reference_roles']['audition_text'] if preparing else context['plan']['request']['text']
        result = execute(spec, text, context['execution']['bundles'],
                         context['execution']['references'], reference_temp_dir=Path(staging)/stage/'conditioning',
                         backend_factory=backend_factory, managed_owner_uid=10001, reference_fd=config.get('reference_fd'))
    else:
        result = execute(spec, context['plan']['request']['text'], context['execution']['bundles'], backend_factory=backend_factory)
    if time.monotonic() >= cutoff: raise Refused('expired_result')
    stage = attempt['staging_id']
    if Path(stage).name != stage or stage in ('.', '..'): raise Refused('invalid_stage')
    preparing = context['plan']['operation'] == 'voice_prepare'
    files = [('reference.wav', result['reference']['data']), ('audition.wav', result['audition']['data'])] if preparing else [('output.wav', result['data'])]
    directory = os.open(Path(staging)/stage, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for name, data in files:
            fd = os.open(name, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o640, dir_fd=directory)
            with os.fdopen(fd, 'wb') as output:
                output.write(data); output.flush(); os.fsync(output.fileno())
    finally:
        os.close(directory)
    if not preparing:
        asset = dict(id='asset-'+uuid4().hex, sha256=result['sha256'], frames=result['frames'], sample_rate=spec.sample_rate, channels=1, media_format='wav', media_subtype='PCM_16', provenance=dict(kind='generated', source_sha256=result['sha256'], source_id=generated_source(context), missing_fields=[]))
    manifest = {k:message[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')}
    if preparing:
        from worker.engines.pocket_prepare import outputs
        from worker.synthetic import candidates
        assets = outputs(context, result)
        proposed_candidates = candidates(message, assets[1], assets[0])
    else:
        assets = [asset]; proposed_candidates = []
    manifest.update(operation=context['plan']['operation'], artifact_id='artifact-'+uuid4().hex, outputs=assets, candidates=proposed_candidates)
    message.update(kind='result', stage='completed', result_asset_ids=[a['id'] for a in assets])
    send(connection, dict(message=message, manifest=manifest), cutoff)


# Retain the existing private test/integration entry point.
run_kokoro = run_generation


def run_export(connection, config, staging, cutoff):
    from uuid import uuid4
    import hashlib
    from worker.transport import Session, authenticate, canonical, send, Refused
    from contracts import validate as oracle
    from worker.audio_export import execute, provenance
    job,context=config['job'],config['context']
    attempt=job['attempts'][-1]
    binding=hashlib.sha256(canonical([job,context,config['allowed_assets'],config['profiles']])).hexdigest()
    authenticate(connection,Session(config['secret'],binding,config['control_uid'],config['gid']),cutoff,control=False)
    message=receive(connection,cutoff)
    oracle.worker_message(message,job,config['allowed_assets'],transport_owner=attempt['owner_id'],context=context,current_profiles=config['profiles'])
    if message['kind']!='execute': raise Refused('unexpected_message')
    stage=attempt['staging_id']
    if Path(stage).name!=stage or stage in ('.','..'): raise Refused('invalid_stage')
    data,measured=execute(context,config['reference_fd'],Path(staging)/stage,image_digest=config['converter_image_digest'],timeout=max(.01,cutoff-time.monotonic()))
    if time.monotonic()>=cutoff: raise Refused('expired_result')
    asset=dict(id='asset-'+uuid4().hex,**measured,provenance=provenance(context['plan']))
    manifest={k:message[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')}
    a=context['plan']['assets'][0]
    manifest.update(operation='audio_export',artifact_id='artifact-'+uuid4().hex,outputs=[asset],candidates=[],audio_export=dict(source_asset_id=a['id'],source_sha256=a['sha256'],settings=context['plan']['request'],converter=context['plan']['resolved']['identity']['converter']))
    message.update(kind='result',stage='completed',result_asset_ids=[asset['id']])
    send(connection,dict(message=message,manifest=manifest),cutoff)


def contain_kokoro(context):
    """Linux child process ceilings; container-wide topology remains install-owned."""
    import resource
    from worker.runtime_protocol import execution
    spec = execution(context)
    if spec is None: raise ValueError('execution_required')
    resource.setrlimit(resource.RLIMIT_AS, (spec.memory_bytes, spec.memory_bytes))
    file_limit = max(spec.max_output_bytes, getattr(spec, 'reference_bytes', 0))
    if spec.engine == 'kokoro':
        # Linux applies FSIZE to anonymous memfd writes too. eSpeak isolates its
        # trusted library there; the engine independently measures/rejects WAVs
        # above max_output_bytes before staging or publication.
        from lib.tts_backends.espeak_memfd import MAX_LIBRARY_BYTES
        file_limit = max(file_limit, MAX_LIBRARY_BYTES)
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
    resource.setrlimit(resource.RLIMIT_NPROC, (48, 48))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(spec.threads)


def main():
    connection = socket.socket(fileno=int(sys.argv[1]))
    config = receive(connection, time.monotonic() + 2)
    mode = config.pop('fixture')
    protected_stages = config.pop('protected_stages', [])
    if mode in ('busy', 'descendants'):
        leader = os.getpid()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if mode == 'descendants':
            if os.fork() == 0:
                os.setsid()
                if os.fork() != 0:
                    os._exit(0)
        if os.getpid() == leader:
            Path('/staging/'+config['job']['attempts'][-1]['staging_id']+'/ready').write_text('synthetic_fault_ready')
        while True:
            pass
    if mode == 'permissions':
        status = Path('/proc/self/status').read_text()
        assert 'CapEff:\t0000000000000000' in status
        assert 'CapPrm:\t0000000000000000' in status
        assert 'NoNewPrivs:\t1' in status
        assert os.getgroups() == []
        for stage in protected_stages:
            try:
                os.rename('/staging/'+stage, '/staging/'+stage+'-forbidden')
            except PermissionError:
                pass
            else:
                raise RuntimeError('other_attempt_rename_allowed')
            try:
                open('/staging/'+stage+'/output.wav', 'rb').close()
            except (PermissionError, FileNotFoundError):
                continue
            raise RuntimeError('other_attempt_accessible')
        with socket.socket(socket.AF_UNIX) as probe:
            try:
                probe.connect('/ipc/lifecycle.sock')
            except PermissionError:
                pass
            else:
                raise RuntimeError('management_socket_accessible')
        for path in ('/ipc/lifecycle.key', '/ipc/execution.key', '/receipts/journal.json', '/control/owner.db', '/publication'):
            try:
                open(path, 'rb').close()
            except (PermissionError, FileNotFoundError):
                continue
            raise RuntimeError('protected_path_accessible')
        try:
            os.kill(1, 0)
        except PermissionError:
            pass
        else:
            raise RuntimeError('supervisor_signal_allowed')
        assert not any(int(p.name) > 2 and int(p.name) != connection.fileno() for p in Path('/proc/self/fd').iterdir() if p.name.isdigit() and Path(p).exists())
    if mode in ('kokoro','pocket','qwen','audio_export'):
        from worker.runtime_protocol import execution, dispatch_mode
        if dispatch_mode(execution(config['context'])) != mode: raise ValueError('execution_dispatch_mismatch')
        contain_kokoro(config['context'])
        if mode=='audio_export':
            run_export(connection,config,'/staging',time.monotonic()+min(25,(datetime.fromisoformat(config['job']['deadline'])-datetime.now(timezone.utc)).total_seconds()))
        else: run_generation(connection, config, '/staging', time.monotonic()+30)
    else:
        from worker.runtime_protocol import execution
        if execution(config['context']) is not None: raise ValueError('execution_dispatch_mismatch')
        run(connection, config, '/staging', time.monotonic() + 30)
    names = ('reference.wav','audition.wav') if config['context']['plan'].get('reference_roles') else ('output.wav',)
    for name in names:
        os.chmod('/staging/'+config['job']['attempts'][-1]['staging_id']+'/'+name, 0o644)


if __name__ == '__main__':
    main()
