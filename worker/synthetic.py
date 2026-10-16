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
        if message['kind']!='execute' or message['plan']['operation']!='speech_generate':
            raise ValueError('unsupported_operation')
        time.sleep(min(30,max(0,float(delay))))
        buffer=io.BytesIO()
        with wave.open(buffer,'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(48000)
            wav.writeframes(b'\0'*9600)
        data=buffer.getvalue();digest=hashlib.sha256(data).hexdigest()
        with open(Path(directory)/'output.wav','xb') as output:
            output.write(data);output.flush();os.fsync(output.fileno())
        asset=dict(id='asset-'+uuid4().hex,sha256=digest,frames=4800,sample_rate=48000,channels=1,media_format='wav',media_subtype='PCM_16',provenance=dict(kind='synthetic',source_sha256=digest,source_id='phase-1a-silence',missing_fields=[]))
        manifest={k:message[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')}
        manifest.update(operation='speech_generate',artifact_id='artifact-'+uuid4().hex,outputs=[asset],candidates=[])
        message.update(kind='result',stage='completed',result_asset_ids=[asset['id']])
        connection.sendall(json.dumps(dict(message=message,manifest=manifest),separators=(',',':')).encode())

if __name__=='__main__': main()
