"""Generated preset configuration: no fixture identities or qualification claims."""
import copy
import hashlib
import json
from pathlib import Path
from contracts import validate as oracle
from lib.spoken_text import freeze
from service.provisioning import identity,manifest_sha256,validate_expected
from worker.engines.execution import Execution
from worker.engines.native_kokoro import NativeKokoroExecution,PROFILE,MEMORY_BYTES,JOB_SECONDS
from worker import native_cpu_protocol as p

DESCRIPTOR_SHA256='afbc1fafed69c55adff4e94ccdb936f347f99cdc60171ed0ae9a2314b080c7f3'
TEXT='Hello. The test is complete.'
VOICE='kokoro.en.af_heart'
ALIAS='default.narration'


def expected():
    path=Path(__file__).resolve().parents[1]/'requirements/tts/kokoro-en-macos.expected.json'
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=DESCRIPTOR_SHA256:raise ValueError('native_cpu_descriptor_changed')
    return validate_expected(json.loads(raw))


def profile(expected,*,revision,installed_at):
    ident=dict(engine='kokoro',model=expected['checkpoint'],bundle_sha256=manifest_sha256(expected),device='cpu',precision='float32')
    result=dict(id=PROFILE,device='cpu',topology='packaged_native',support='unqualified',installation='installed',
        configuration='valid',readiness='not_ready',verification='synthetic',checked_at=installed_at,
        qualification_receipt='',config_revision=revision,
        next_action='Run an explicit bounded qualification session; installation does not establish readiness.',
        compatibility=[dict(operation='speech_generate',mode='kokoro_preset',identity=ident)],
        limits=dict(expensive_slots=1,resident_models=1,threads=2,queue=2,upload_bytes=1048576,decoded_ticks=960000,
            text_chars=256,variants=1,disk_bytes=16*1024**2,deadline_seconds=JOB_SECONDS,ram_bytes=MEMORY_BYTES,vram_bytes=0))
    oracle.validate('profile',result)
    return result


def context(expected,profile,native,inventory):
    ident=copy.deepcopy(profile['compatibility'][0]['identity'])
    settings=dict(kind='kokoro_preset',language='en');settings_hash=oracle.canonical_hash(settings)
    version=dict(id='model-preset-'+manifest_sha256(expected),voice_id=VOICE,origin='preset',reference_asset_ids=[],reference_asset_sha256={},reference_text='',
        identity=ident,settings=settings,settings_sha256=settings_hash,
        provenance=dict(kind='generated',source_sha256=manifest_sha256(expected),source_id='model-preset:af_heart',missing_fields=[]),
        configuration=dict(mode='kokoro_preset',preset='af_heart',language='en'),design_lineage=None)
    request=dict(voice_version_id=version['id'],text=TEXT,identity=ident,fallback='forbid',variants=1,reuse='never')
    resolved=dict(reuse_configuration=None,reuse_settings=None,reuse_identity=None,kind='speech',
        identity=dict(requested=ident,effective=ident,fallback='forbid',reason='',profile_id=PROFILE),
        profile=oracle.profile_snapshot(profile),operator_fallback='forbid',settings=settings,settings_sha256=settings_hash)
    plan=dict(schema_version='1.3.0',operation='speech_generate',request=request,project=None,voice_versions=[version],assets=[],resolved=resolved)
    freeze(plan)
    bundles=[dict(instructions=False,engine='kokoro',model=expected['checkpoint'],mode='kokoro_preset',bundle_sha256=manifest_sha256(expected),languages=['en'],selectors=[dict(value='af_heart',languages=['en'])])]
    model=Execution(identity(expected),manifest_sha256(expected),expected['checkpoint'],expected['revision'],'af_heart','en',2,256,480000,960044,MEMORY_BYTES)
    spec=NativeKokoroExecution(model,VOICE,version['id'])
    result=dict(plan=plan,bundles=bundles,execution=dict(descriptor=spec.identity(),
        bundles={spec.bundle_id:'/native-models/bundles/'+spec.bundle_id},native=copy.deepcopy(native),hardware=p.inventory_wire(inventory)))
    oracle.frozen_plan(plan,bundles);p.native_execution(result)
    return result


def template(context,profiles,principal):
    """Dormant trusted selector template, never a submitted/running model job."""
    from datetime import datetime,timezone,timedelta
    job=dict(id='native-preset-template',operation='speech_generate',input_sha256=oracle.plan_hash(context['plan']),principal_id=principal,
        idempotency_key='native-preset-template',state='running',generation=1,artifact_ids=[],stage='loading',
        deadline=(datetime.now(timezone.utc)+timedelta(seconds=JOB_SECONDS)).isoformat(),
        attempts=[dict(id='template-attempt',generation=1,owner_id='template-owner',staging_id='template-staging',reaped=False)])
    oracle.validate('job',job);oracle.stored_context(job,context)
    return [job,context,[],profiles]
