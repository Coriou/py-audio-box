"""One explicit real public API job; no network acquisition or automatic retry.

ASGI transport exercises the installed authenticated HTTP routes in process. The
actual Dispatcher/Core and mTLS model custodian are unchanged. Ordinary serve's
TCP listener is qualified separately; this command does not benchmark the GPU.
"""
import asyncio
import base64
from datetime import datetime,timezone
import hashlib
import secrets
import time
from urllib.parse import quote

from contracts import validate as oracle
from worker.transport import Refused,canonical


async def run(root):
    import httpx
    from service import native_package as files
    from service.catalogue import OPERATIONS
    from service.native_cpu_configuration import ALIAS,TEXT
    from service.native_cpu_setup import create_app
    from service.native_cpu_measurement import binding,retained,wav,publish,VERSION,require
    started=time.monotonic();deadline=started+90
    app=create_app(root,_selftest=True)
    prepared=None
    async with app.router.lifespan_context(app):
        prepared=app.state.installed_lifecycle['notification']
        credential=files.document(prepared.root/'credential.json')['credentials'][0]['token']
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://native-selftest',
                headers={'Authorization':'Bearer '+credential},trust_env=False,follow_redirects=False) as client:
            async def call(name,payload,key=None):
                spec=OPERATIONS[name];oracle.validate(spec['request'],payload)
                route=spec['http']['route'];query=dict(payload)
                for field in tuple(query):
                    marker='{'+field+'}'
                    if marker in route:route=route.replace(marker,quote(str(query.pop(field)),safe=''))
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('native_cpu_selftest_public_deadline; outcome may be unknown; no automatic retry')
                request=(client.post(route,content=canonical(payload),headers={'Idempotency-Key':key,'Content-Type':'application/json'})
                    if spec['http']['method']=='POST' else client.get(route,params=query))
                response=await asyncio.wait_for(request,remaining)
                if response.status_code!=200:raise Refused('native_cpu_selftest_public_'+str(response.status_code)+'_'+name)
                return oracle.validate(spec['response'],response.json())
            selected=await call('voice_resolve',dict(name=ALIAS))
            version=await call('voice_version_get',dict(voice_version_id=selected['version_ids'][-1]))
            submitted=await call('speech_generate',dict(voice_version_id=version['id'],text=TEXT,identity=version['identity'],fallback='forbid',variants=1,reuse='never'),key='native-selftest-'+secrets.token_hex(12))
            cursor=''
            while True:
                result=await call('job_wait',dict(job_id=submitted['job_id'],cursor=cursor,wait_seconds=1))
                job=result['job'];cursor=result['next_cursor']
                if job['state'] in ('succeeded','failed','cancelled'):break
            require(job['state']=='succeeded' and len(job['artifact_ids'])==1)
            artifact=await call('artifact_get',dict(artifact_id=job['artifact_ids'][0]));require(len(artifact['asset_ids'])==1)
            asset_id=artifact['asset_ids'][0]
            metadata=await call('asset_get',dict(asset_id=asset_id))
            # Existing range API, at most 32x32KiB; whole hash after byte transfer.
            first=await call('asset_content',dict(asset_id=asset_id,start=0,end_exclusive=1))
            total=first['total_bytes'];require(0<total<=960044 and first['content_sha256']==metadata['sha256'])
            raw=bytearray(base64.b64decode(first['content_base64'],validate=True));require(len(raw)==1 and first['start']==0 and first['end_exclusive']==1)
            for _ in range(32):
                content=await call('asset_content',dict(asset_id=asset_id,start=len(raw),end_exclusive=min(total,len(raw)+32768)))
                data=base64.b64decode(content['content_base64'],validate=True)
                require(content['start']==len(raw) and content['end_exclusive']==len(raw)+len(data) and content['content_sha256']==metadata['sha256'])
                require(total==content['total_bytes'] and 0<total<=960044)
                raw.extend(data)
                if len(raw)==total:break
                require(bool(data) and len(raw)<total)
            require(len(raw)==total);media=wav(bytes(raw));require(media['sha256']==metadata['sha256'])
            # A public succeeded job can precede ACK. Wait for ACK on the actual
            # owner journal without reading SQLite on a foreign live thread.
            while True:
                records=prepared.supervisor.journal.value['records']
                matches=[r for r in records if r['snapshot'][0]['id']==job['id']]
                if len(matches)==1 and matches[0]['phase']=='terminal' and matches[0]['acked']:break
                if time.monotonic()>=deadline:raise TimeoutError('native_cpu_selftest_ack_deadline')
                await asyncio.sleep(.02)
            measured_binding=binding(prepared)
            elapsed=time.monotonic()-started
    lifecycle=app.state.installed_lifecycle
    require(lifecycle['closed'] is True and lifecycle['error'] is None and prepared.closed)
    proof=retained(prepared.root,job['id'],configuration=prepared.identity,runtime_manifest=prepared.runtime_manifest,descriptor=prepared.descriptor,baseline=prepared.baseline_profile)
    require(proof['wav']==media)
    value=dict(version=VERSION,binding=measured_binding,proof=proof,elapsed_seconds=elapsed,
        checked_at=datetime.now(timezone.utc).isoformat(),cleanup='owned_lifecycle_closed',
        session_sha256=oracle.canonical_hash(files.document(prepared.root/'measurement-intent.json')))
    publish(prepared,value)
    return dict(qualification='local_selftest',ready_for_speech=True,job_id=job['id'],wav=media,
        cold_end_to_end_seconds=elapsed,semantic_quality_verified=False,retained_history_jobs=63)


def command(root):
    # Like setup/serve, a requested TERM must unwind the owned lifecycle rather
    # than terminate the control process before its model has been reaped.
    import signal
    def interrupted(number,frame):raise KeyboardInterrupt('Native selftest interrupted; retain state, no automatic retry.')
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:return asyncio.run(run(root))
    finally:signal.signal(signal.SIGTERM,previous)
