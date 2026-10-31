import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contracts.validate import validate
from service.catalogue import FIXTURES, IMPLEMENTED, OPERATIONS, VOICE_READS, PRODUCTION_OPERATIONS
from clients.http import HTTP


def cli(url, *args):
    return subprocess.run([sys.executable,'-m','clients.cli','--url',url,*args], capture_output=True,text=True,timeout=15)


def test_http_cli_mcp_parity_and_errors(configured_url, tmp_path):
    url = configured_url
    expected = HTTP(url).call('discover', {})
    result = cli(url,'discover')
    assert result.returncode == 0 and json.loads(result.stdout) == expected
    assert expected['operations'] == list(IMPLEMENTED) and expected['engines'] == [] and expected['profiles'] == []
    assert HTTP(url).get('/v1/status')['ready_for_speech'] is False
    for route in ('/v1/status', '/skeleton'):
        assert httpx.get(url+route+'?scan=true',trust_env=False).status_code == 422
    for operation in OPERATIONS:
        detail = HTTP(url).get('/skeleton/operations/'+operation)
        assert detail['operation'] == OPERATIONS[operation]
        assert detail['availability'] == ('implemented_fixture' if operation in IMPLEMENTED else 'unavailable')
    invalid = httpx.get(url+'/v1/voices?limit=21',trust_env=False)
    assert invalid.status_code == 422
    validate('error',invalid.json())
    invalid_cli = cli(url,'voices','list','--limit','21')
    assert invalid_cli.returncode == 2 and json.loads(invalid_cli.stdout)['code'] == 'invalid_request'
    for path in ['/v1/capabilities?scan=true','/v1/assets/impulse?extra=1','/v1/assets/impulse/content?start=0&end_exclusive=0','/v1/assets/nope']:
        response = httpx.get(url+path,trust_env=False)
        assert response.status_code in (404,422)
        validate('error',response.json())
    response = httpx.post(url+'/v1/operations/speech_generate',json={},trust_env=False)
    assert response.status_code == 404
    assert httpx.get(url+'/skeleton/operations/speech_generate',trust_env=False).json()['availability'] == 'unavailable'

    async def check():
        params = StdioServerParameters(command=sys.executable,args=['-m','clients.mcp'],env={**os.environ,'SPEECH_URL':url})
        async with stdio_client(params) as (reader,writer):
            async with ClientSession(reader,writer) as session:
                init = await session.initialize()
                assert init.protocolVersion
                assert init.serverInfo.name == 'py-audio-box'
                tools = (await session.list_tools()).tools
                assert {t.name for t in tools} == {OPERATIONS[o]['mcp'] for o in (*IMPLEMENTED,*VOICE_READS,*PRODUCTION_OPERATIONS)} | {'speech_enroll','speech_speak','speech_status','speech_operation_detail'}
                assert 'speech_skeleton_status' not in {t.name for t in tools}
                status = HTTP(url).get('/v1/status')
                assert status == HTTP(url).get('/skeleton')
                status_cli = cli(url, 'status')
                assert status_cli.returncode == 0 and json.loads(status_cli.stdout) == status
                for spelling in ('speech_status', 'speech_skeleton_status'):
                    # Exercise SDK dispatch, including hidden aliases with input validation disabled.
                    response = await asyncio.wait_for(session.call_tool(spelling, {}), 10)
                    assert not response.isError and response.structuredContent == status
                    response = await asyncio.wait_for(session.call_tool(spelling, {'scan': True}), 10)
                    assert response.isError and response.structuredContent['code'] == 'invalid_request'
                discovered = await session.call_tool('speech_discover',{})
                assert discovered.structuredContent == expected and not discovered.isError
                bad = await session.call_tool('speech_discover',{'scan':True})
                assert bad.isError and bad.structuredContent['code'] == 'invalid_request'
                missing = await session.call_tool('speech_asset_get',{'asset_id':'absent'})
                assert missing.isError and missing.structuredContent == HTTP_error(url)
                cap = await session.call_tool('speech_asset_content',{'asset_id':'impulse','start':0,'end_exclusive':3073})
                assert cap.isError and cap.structuredContent['code'] == 'resource_limit'
                good = await session.call_tool('speech_asset_content',{'asset_id':'impulse','start':0,'end_exclusive':3072})
                assert not good.isError and len(base64.b64decode(good.structuredContent['content_base64'])) == 3072
                for name in ['speech_asset_upload']:
                    absent = await session.call_tool(name,{})
                    assert absent.isError and absent.structuredContent['code'] == 'not_found'
                malformed = await session.call_tool('speech_speech_generate',{})
                assert malformed.isError and malformed.structuredContent['code'] == 'invalid_request'
                detail = await session.call_tool('speech_operation_detail',{'operation_id':'speech_generate'})
                assert detail.structuredContent['availability'] == 'unavailable'
    asyncio.run(check())


def HTTP_error(url):
    return httpx.get(url+'/v1/assets/absent',trust_env=False).json()


def test_host_transfer_keeps_bytes_out_of_summary(server_url,tmp_path):
    target = tmp_path/'result.wav'
    result = cli(server_url,'asset','content','impulse','--output',str(target))
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary['bytes'] > 3072 and 'content_base64' not in summary and len(result.stdout.encode()) < 2048
    assert target.read_bytes() == (FIXTURES/'impulse.wav').read_bytes()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == summary['sha256']
    target.write_bytes(b'preserve')
    retry = cli(server_url,'asset','content','impulse','--output',str(target))
    assert retry.returncode == 2 and target.read_bytes() == b'preserve'
    assert list(tmp_path.iterdir()) == [target]


def test_discovery_without_ml_scan_or_network():
    # A fresh interpreter forbids network and runtime model/file enumeration.
    code = '''
import sys, socket, pathlib
class Block:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch','transformers','numpy','soundfile','lib','apps'}:
   raise AssertionError('forbidden ML/legacy import: '+fullname)
sys.meta_path.insert(0,Block())
def denied(*a,**kw): raise AssertionError('discovery attempted scan/network')
socket.socket.connect = denied
pathlib.Path.glob = denied
pathlib.Path.rglob = denied
from service.api import discover, EmptyQuery, summary
from service.catalogue import operation_detail
assert discover(EmptyQuery())['engines'] == []
assert not summary()['ready_for_speech']
assert operation_detail('speech_generate')['availability'] == 'unavailable'
'''
    probe = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=15)
    assert probe.returncode == 0, probe.stderr


def test_optional_upload_discovery_is_bounded(monkeypatch):
    from clients.mcp import tools
    from service.catalogue import compact
    from tests.release.measure import measure
    monkeypatch.delenv('SPEECH_UPLOAD_ROOT', raising=False)
    baseline = tools()
    default = measure()
    assert default['initial_connection_tokens'] <= 3500
    assert default['metrics']['skeleton_status']['utf8_bytes'] <= 2048
    monkeypatch.setenv('SPEECH_UPLOAD_ROOT', '/private/operator-authorized')
    optional = tools()
    assert len(optional) == len(baseline) + 1
    extra = next(t for t in optional if t.name == 'speech_asset_upload_file')
    assert [t for t in optional if t.name != extra.name] == baseline
    assert len(compact(extra.model_dump(mode='json', exclude_none=True)).encode()) <= 1024
    # A single finite opt-in tool adds no status aliases or audio payload.
    assert 'speech_skeleton_status' not in {t.name for t in optional}
    assert measure()['initial_connection_tokens'] <= default['initial_connection_tokens'] + 256
