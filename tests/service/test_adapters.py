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
from service.catalogue import FIXTURES, IMPLEMENTED, OPERATIONS
from clients.http import HTTP


def cli(url, *args):
    return subprocess.run([sys.executable,'-m','clients.cli','--url',url,*args], capture_output=True,text=True,timeout=15)


def test_http_cli_mcp_parity_and_errors(configured_url, tmp_path):
    url = configured_url
    expected = HTTP(url).call('discover', {})
    result = cli(url,'discover')
    assert result.returncode == 0 and json.loads(result.stdout) == expected
    assert expected['operations'] == list(IMPLEMENTED) and expected['engines'] == [] and expected['profiles'] == []
    assert HTTP(url).get('/skeleton')['ready_for_speech'] is False
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
                tools = (await session.list_tools()).tools
                assert len(tools) == 6
                assert not any(t.name == 'speech_speech_generate' for t in tools)
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
                for name in ['speech_asset_upload','speech_speech_generate']:
                    absent = await session.call_tool(name,{})
                    assert absent.isError and absent.structuredContent['code'] == 'not_found'
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
