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


def candidates(message, asset):
    """Exact synthetic prepare proposal; never caller-provided candidate metadata."""
    if message['plan']['operation'] != 'voice_prepare': return []
    import copy
    from contracts import validate as oracle
    plan = message['plan']; resolved = plan['resolved']
    version = dict(id='prospective-'+message['job_id'], voice_id='synthetic-voice',
                   origin='clone', reference_asset_ids=[asset['id']],
                   reference_text=plan['request']['reference_text'],
                   identity=copy.deepcopy(resolved['identity']['effective']),
                   settings=copy.deepcopy(resolved['reuse_settings']),
                   settings_sha256=oracle.canonical_hash(resolved['reuse_settings']),
                   provenance=copy.deepcopy(asset['provenance']),
                   configuration=copy.deepcopy(resolved['reuse_configuration']), design_lineage=None)
    return [dict(schema_version='1.3.0', id='candidate-'+message['job_id'],
                 job_id=message['job_id'], version=version, audition_asset_id=asset['id'])]


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
        buffer=io.BytesIO()
        with wave.open(buffer,'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(48000)
            wav.writeframes(b'\0'*9600)
        data=buffer.getvalue();digest=hashlib.sha256(data).hexdigest()
        with open(Path(directory)/'output.wav','xb') as output:
            output.write(data);output.flush();os.fsync(output.fileno())
        asset=dict(id=(message['plan']['resolved']['reuse_configuration']['reference_asset_id'] if message['plan']['operation']=='voice_prepare' else 'asset-'+uuid4().hex),sha256=digest,frames=4800,sample_rate=48000,channels=1,media_format='wav',media_subtype='PCM_16',provenance=dict(kind='synthetic',source_sha256=digest,source_id='phase-1a-silence',missing_fields=[]))
        manifest={k:message[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')}
        manifest.update(operation=message['plan']['operation'],artifact_id='artifact-'+uuid4().hex,outputs=[asset],candidates=candidates(message,asset))
        message.update(kind='result',stage='completed',result_asset_ids=[asset['id']])
        connection.sendall(json.dumps(dict(message=message,manifest=manifest),separators=(',',':')).encode())

if __name__=='__main__': main()
