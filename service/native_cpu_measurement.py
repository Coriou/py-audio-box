"""Owner-local functional evidence, corroborated against retained public proofs.

This is drift protection, not authentication against the installation's owner.
No request payload, configuration flag or raw backend result can issue readiness.
"""
import copy
from datetime import datetime,timezone
import hashlib
import io
import json
import math
import os
import platform
import sqlite3
import wave

from contracts import validate as oracle
from service import native_package as files
from service.native_runtime import _read,_sync
from service.worker_transport import Authority
from worker import native_cpu_protocol as p
from worker.transport import Refused,canonical

VERSION='native-cpu-local-measurement-v1'
ACTION='Run doctor --selftest explicitly on this installation; retain existing state.'


def require(value):
    if not value:raise Refused('native_cpu_measurement_invalid: '+ACTION)


def binding(prepared):
    from worker.native_cpu_qualification import boot_identity
    from service.provisioning import manifest_sha256
    return dict(configuration=prepared.identity,runtime_manifest=prepared.runtime_manifest,
        profile=oracle.canonical_hash(prepared.baseline_profile),model=manifest_sha256(prepared.descriptor),
        host=dict(system=platform.system(),machine=platform.machine(),os=platform.mac_ver()[0],boot=boot_identity()))


def wav(raw):
    import array
    with wave.open(io.BytesIO(raw),'rb') as stream:
        require((stream.getnchannels(),stream.getsampwidth(),stream.getframerate(),stream.getcomptype())==(1,2,24000,'NONE'))
        frames=stream.getnframes();require(0<frames<=480000)
        samples=stream.readframes(frames);require(len(samples)==frames*2 and not stream.readframes(1))
    require(any(array.array('h',samples)))
    return dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),frames=frames,rate=24000,duration=frames/24000)


def retained(root,job_id,*,configuration,runtime_manifest,descriptor,baseline):
    """Read-only selected proof; full bounded history is also checked at activation.

    Caller holds the installation owner lock. Never opens/repairs Core as a writer.
    """
    from service.provisioning import manifest_sha256
    from worker.native_child import validate_proposal
    from service.native_cpu_configuration import TEXT
    root=files.directory(str(root));control=root/'state'/'control'
    # Validate path ownership/no-follow before SQLite's read-only connection.
    _read(control/'core.sqlite',64*1024**2)
    with sqlite3.connect((control/'core.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        def row(table,key,value):
            rows=db.execute('SELECT * FROM '+table+' WHERE '+key+'=?',(value,)).fetchall()
            require(len(rows)==1);return dict(rows[0])
        stored=row('jobs','id',job_id);job=json.loads(stored['job']);context=json.loads(stored['context'])
        dispatch=row('dispatches','job_id',job_id);publication=row('publications','job_id',job_id)
        authority=Authority(*json.loads(dispatch['authority']))
        require(dispatch['authority'].encode()==authority._snapshot and authority.binding==dispatch['binding'])
        require(canonical(context)==canonical(authority.context) and job['id']==authority.job['id'])
        require(job['state']=='succeeded' and job['stage']=='completed' and len(job['attempts'])==1 and job['attempts'][0]['reaped'] is True)
        require(dispatch['acked']==1 and dispatch['phase']=='acknowledged' and canonical(json.loads(dispatch['outcome']))==canonical(job))
        spec=p.native_execution(context);native=p.native_binding(context)
        require(native['configuration_sha256']==configuration and native['runtime_manifest_sha256']==runtime_manifest)
        require(spec.manifest_sha256==manifest_sha256(descriptor) and context['plan']['request']['text']==TEXT)
        require(context['plan']['resolved']['profile']==oracle.profile_snapshot(baseline))
        receipt=json.loads(dispatch['receipt'])
        p.terminal_receipt(receipt,authority=authority,runtime_id=dispatch['runtime'],local_import_uid=os.getuid())
        require(receipt['timely'] is True and receipt['reaped'] is True)
        artifact=json.loads(publication['artifact']);manifest=json.loads(publication['manifest']);message=json.loads(publication['message'])
        require(len(manifest['outputs'])==1 and job['artifact_ids']==[artifact['id']])
        output=manifest['outputs'][0];raw=_read(control/'assets'/output['id'],1048576)
        media=wav(raw)
        oracle.publication(authority.job,authority.job['attempts'][0],artifact,{output['id']:media['sha256']},
            transport_owner=authority.job['attempts'][0]['owner_id'],context=context,manifest=manifest,
            verified_result=message,allowed_assets=authority.allowed_assets)
        retained_artifact=row('artifacts','id',artifact['id']);require(canonical(json.loads(retained_artifact['metadata']))==canonical(artifact))
        ops=[r[0] for r in db.execute('SELECT op FROM dispatch_commands WHERE dispatch_id=? ORDER BY seq',(dispatch['seq'],))]
        require(ops[:2]==['prepare','start'] and ops.count('collect')==1)
        collect=ops.index('collect');require(collect>=3 and ops[collect-1]=='observe' and all(op in ('observe','stop') for op in ops[2:collect]) and bool(ops[collect+1:]) and all(op=='ack' for op in ops[collect+1:]))
    journal=files.document(root/'state'/'worker'/'native.json')
    require(set(journal)=={'payload','sha256'} and oracle.canonical_hash(journal['payload'])==journal['sha256'])
    records=[r for r in journal['payload']['records'] if r['id']==dispatch['runtime']]
    require(len(records)==1);record=records[0]
    require(record['phase']=='terminal' and record['acked'] is True and canonical(record['snapshot'])==authority._snapshot and canonical(record['receipt'])==canonical(receipt))
    require(record['output']==dict(sha256=media['sha256'],bytes=media['bytes']))
    require(canonical(record['proposal'])==canonical(dict(message=message,manifest=manifest)))
    validate_proposal(record['snapshot'],record['proposal'],raw,spec,protocol=p)
    return dict(job_id=job_id,binding=authority.binding,runtime_id=dispatch['runtime'],
        terminal_sha256=oracle.canonical_hash(receipt),sampled_resources=copy.deepcopy(receipt['resources']),publication_sha256=oracle.canonical_hash(dict(artifact=artifact,manifest=manifest,message=message)),wav=media)


def validate(value,prepared):
    p._closed(value,('version','binding','proof','elapsed_seconds','checked_at','cleanup','session_sha256'))
    require(value['version']==VERSION and value['binding']==binding(prepared) and value['cleanup']=='owned_lifecycle_closed')
    require(value['session_sha256']==oracle.canonical_hash(files.document(prepared.root/'measurement-intent.json')))
    seconds=value['elapsed_seconds'];require(type(seconds) in (int,float) and math.isfinite(seconds) and 0<seconds<=90)
    checked=datetime.fromisoformat(value['checked_at']);require(checked.tzinfo is not None and checked<=datetime.now(timezone.utc))
    require(type(value['proof']) is dict and type(value['proof'].get('job_id')) is str)
    proof=retained(prepared.root,value['proof']['job_id'],configuration=prepared.identity,runtime_manifest=prepared.runtime_manifest,
        descriptor=prepared.descriptor,baseline=prepared.baseline_profile)
    require(canonical(proof)==canonical(value['proof']))
    result=copy.deepcopy(prepared.baseline_profile)
    result.update(support='qualified',readiness='ready',verification='measured',checked_at=value['checked_at'],
        qualification_receipt='local-selftest:'+oracle.canonical_hash(value),
        next_action='Local EN af_heart functional preview; revalidate after reboot/drift. Retained history is limited to 63 jobs; no rollover.')
    oracle.validate('profile',result);return result


def load(prepared):
    path=prepared.root/'measurement.json'
    if not path.exists() and not path.is_symlink():return None
    try:return validate(files.document(path),prepared)
    except (OSError,ValueError,KeyError,TypeError,sqlite3.Error) as exc:
        raise Refused('native_cpu_measurement_revalidation_required: '+ACTION) from exc


def publish(prepared,value):
    """Only after the actual app owner/custodian closed; never accept caller JSON."""
    require(prepared.closed and prepared.supervisor.active is None and not prepared.thread.is_alive() and not prepared.errors)
    # Reacquire the same exclusion before corroboration/publication. No session,
    # history, aliases, artifacts or model files are reset or deleted here.
    import fcntl
    lock=os.open(prepared.root/'owner.lock',os.O_WRONLY|os.O_NOFOLLOW)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def identity(info):return info.st_dev,info.st_ino,info.st_uid,info.st_mode,info.st_nlink
        require(identity(os.fstat(lock))==prepared.owner_identity==identity((prepared.root/'owner.lock').lstat()))
        prepared.revalidate();validate(value,prepared)
        from service.native_cpu_install import write
        temporary=prepared.root/'measurement.pending'
        write(temporary,value)
        require(identity(os.fstat(lock))==prepared.owner_identity==identity((prepared.root/'owner.lock').lstat()))
        os.replace(temporary,prepared.root/'measurement.json');_sync(prepared.root)
    finally:os.close(lock)
