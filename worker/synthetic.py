"""One-shot synthetic worker. Private inherited UDS and owner lock; no store access.

Host subprocess is fault-test evidence only, not a network isolation claim. The
packaged worker remains a separate network_mode:none container (0C probe).
"""
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
from uuid import uuid4
import wave


def candidates(message, asset, reference=None):
    """Exact synthetic prepare proposal; never caller-provided candidate metadata."""
    if message['plan']['operation'] != 'voice_prepare': return []
    import copy
    from contracts import validate as oracle
    plan = message['plan']; resolved = plan['resolved']
    reference = reference or asset
    version = dict(id='prospective-'+message['job_id'], voice_id='synthetic-voice',
                   origin='clone', reference_asset_ids=[reference['id']],
                   reference_text=plan['request']['reference_text'],
                   identity=copy.deepcopy(resolved['identity']['effective']),
                   settings=copy.deepcopy(resolved['reuse_settings']),
                   settings_sha256=oracle.canonical_hash(resolved['reuse_settings']),
                   provenance=copy.deepcopy(reference['provenance']),
                   configuration=copy.deepcopy(resolved['reuse_configuration']), design_lineage=None)
    result = dict(schema_version='1.3.0', id='candidate-'+message['job_id'],
                 job_id=message['job_id'], version=version, audition_asset_id=asset['id'])
    if 'reference_roles' in plan:
        original = next(a for a in plan['assets'] if a['id']==plan['request']['asset_id'])
        result['reference_roles'] = dict(version='reference-roles-v1',
            reference=dict(kind='prepared_reference', asset=copy.deepcopy(reference), original=copy.deepcopy(original),
                in_tick=plan['request']['in_tick'], out_tick=plan['request']['out_tick'], transcript=plan['request']['reference_text']),
            audition=dict(kind='generated_audition', asset=copy.deepcopy(asset), text=plan['reference_roles']['audition_text']))
    return [result]


def fixture_outputs(message):
    """Bounded synthetic PCM fixtures, not source decoding, cloning or speech.

    Reference duration follows exact 48kHz ticks; provenance retains the authorized
    source association. Different PCM markers prevent audition/reference reuse.
    """
    plan=message['plan']; roles=plan.get('reference_roles')
    specs=[('output.wav',4800,0,None)]
    if roles:
        if plan['resolved']['profile']['id'] not in ('cpu-synthetic','cuda-synthetic','metal-synthetic'): raise ValueError('synthetic_profile_required')
        if plan['operation']!='voice_prepare': raise ValueError('synthetic_only')
        ticks=plan['request']['out_tick']-plan['request']['in_tick']
        if not 0<ticks<=480000: raise ValueError('synthetic_segment_limit')
        specs=[('reference.wav',ticks,1,roles['reference_asset_id']),('audition.wav',4800,0,roles['audition_asset_id'])]
    outputs=[];files=[]
    for name,frames,marker,asset_id in specs:
        buffer=io.BytesIO()
        with wave.open(buffer,'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(48000)
            wav.writeframes(bytes([marker,0])*frames)
        data=buffer.getvalue();digest=hashlib.sha256(data).hexdigest()
        provenance=dict(kind='synthetic',source_sha256=digest,source_id='phase-1a-silence',missing_fields=[])
        if name=='reference.wav':
            original=next(a for a in plan['assets'] if a['id']==plan['request']['asset_id'])
            provenance=dict(kind=original['provenance']['kind'],source_sha256=original['sha256'],source_id=original['id'],missing_fields=[])
        asset_id=asset_id or (plan['resolved']['reuse_configuration']['reference_asset_id'] if plan['operation']=='voice_prepare' else 'asset-'+uuid4().hex)
        outputs.append(dict(id=asset_id,sha256=digest,frames=frames,sample_rate=48000,channels=1,media_format='wav',media_subtype='PCM_16',provenance=provenance))
        files.append((name,data))
    return outputs,files


def fixture_manifest(message, outputs):
    manifest={k:message[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')}
    manifest.update(operation=message['plan']['operation'],artifact_id='artifact-'+uuid4().hex,outputs=outputs,
        candidates=candidates(message,outputs[-1],outputs[0]))
    return manifest


def main():
    os.umask(0o077)
    fd, owner_fd, directory, delay = sys.argv[1:]
    # owner_fd stays open until process exit, including on control process death.
    os.fstat(int(owner_fd))
    with socket.socket(fileno=int(fd)) as connection:
        raw=bytearray()
        while not raw.endswith(b'\n') and len(raw)<=65536:
            chunk=connection.recv(65536)
            if not chunk: raise ValueError('control_disconnected')
            raw.extend(chunk)
        if len(raw)>65536: raise ValueError('plan_limit')
        message=json.loads(raw)
        if message['kind']!='execute' or message['plan']['operation'] not in ('speech_generate','voice_prepare'):
            raise ValueError('unsupported_operation')
        time.sleep(min(30,max(0,float(delay))))
        outputs, files = fixture_outputs(message)
        for name, data in files:
            with open(Path(directory)/name,'xb') as output:
                output.write(data);output.flush();os.fsync(output.fileno())
        manifest=fixture_manifest(message, outputs)
        message.update(kind='result',stage='completed',result_asset_ids=[a['id'] for a in outputs])
        connection.sendall(json.dumps(dict(message=message,manifest=manifest),separators=(',',':')).encode())

if __name__=='__main__': main()
