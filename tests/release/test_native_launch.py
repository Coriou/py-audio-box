"""Inert TLS/local ASGI native public launch; no listener or child."""
import asyncio
import threading
import unittest
from unittest.mock import patch
import httpx
from service import native_launch as launch, native_package as n, entrypoint as e
from tests.release.test_native_enrollment import NativeEnrollmentTests
from worker.transport import canonical


class NativeLaunchTests(unittest.TestCase):
    setUp = NativeEnrollmentTests.setUp
    write = NativeEnrollmentTests.write
    enroll = NativeEnrollmentTests.enroll

    def app(self):
        self.enroll()
        self.credential = self.write('credentials', canonical(dict(version=1, credentials=[
            dict(token='x'*64, principal='alice', scopes=['read','production-write'])])))
        return launch.create_native_app(record_file=str(self.record),
            configuration_file=str(self.root/'control'), templates_file=str(self.root/'templates'),
            owner_root=str(self.owner), staging=str(self.f.stage), credential_file=str(self.credential))

    def connector(self):
        original = n.enrolled_runtime
        def runtime(*args, **kwargs):
            kwargs['connector'] = self.f.connect
            result = original(*args, **kwargs)
            self.assertEqual(result.owner, threading.get_ident())
            return result
        return patch.object(n, 'enrolled_runtime', side_effect=runtime)

    def test_authenticated_public_lifespan_and_restart(self):
        app = self.app()
        async def run():
            for _ in range(2):
                async with app.router.lifespan_context(app):
                    owner = app.state.dispatcher
                    self.assertEqual(owner.runtime.owner, owner.thread.ident)
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://inert') as client:
                        self.assertEqual((await client.get('/skeleton')).status_code, 403)
                        self.assertEqual((await client.get('/skeleton',headers={'Authorization':'Bearer wrong'})).status_code,403)
                        headers={'Authorization':'Bearer '+'x'*64}
                        status=(await client.get('/skeleton',headers=headers)).json()
                        self.assertTrue(status['installed']);self.assertTrue(status['activation_ready'])
                        self.assertFalse(status['ready_for_speech']);self.assertFalse(status['synthetic'])
                        caps=(await client.get('/v1/capabilities',headers=headers)).json()
                        self.assertEqual(caps['profiles'],self.templates['profiles'])
                        self.assertEqual(caps['engines'],['qwen'])
                        self.assertNotIn('voice_prepare',caps['operations'])
                        self.assertEqual((await client.get('/v1/voices?limit=20',headers=headers)).status_code,200)
                self.assertFalse(owner.thread.is_alive())
                self.assertTrue(app.state.installed_lifecycle['closed'])
        with self.connector(): asyncio.run(run())

    def test_existing_cli_mcp_same_authenticated_api_and_error(self):
        import contextlib
        import io
        import json
        from clients import cli, mcp
        app=self.app()
        class ASGIClient:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def get(self,path,params=None,headers=None):
                async def request():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://inert') as client:
                        return await client.get(path,params=params,headers=headers)
                return asyncio.run(request())
        def clients():
            with patch('clients.http.httpx.Client',ASGIClient), patch.dict('os.environ',{'SPEECH_CREDENTIAL':'x'*64}):
                output=io.StringIO()
                with contextlib.redirect_stdout(output): self.assertEqual(cli.main(['--url','http://inert','voices','list']),0)
                body=json.loads(output.getvalue())
                response=asyncio.run(mcp.call_tool('speech_voices_list',{'limit':20,'cursor':''}))
                self.assertFalse(response.isError);self.assertEqual(response.structuredContent,body)
                with patch.dict('os.environ',{'SPEECH_CREDENTIAL':'wrong'}):
                    response=asyncio.run(mcp.call_tool('speech_voices_list',{'limit':20,'cursor':''}))
                    self.assertTrue(response.isError);self.assertEqual(response.structuredContent['code'],'forbidden')
        async def run():
            async with app.router.lifespan_context(app): await asyncio.to_thread(clients)
        with self.connector(): asyncio.run(run())

    def test_changed_boot_refuses_public_activation(self):
        app=self.app()
        def mutate(request,response):
            if request['op']=='hello': response['boot']='e'*32
            return response
        self.f.mutate=mutate
        async def run():
            with self.assertRaises(Exception):
                async with app.router.lifespan_context(app): self.fail('boot accepted')
            self.assertTrue(app.state.installed_lifecycle['closed'])
            self.assertFalse((self.owner/'core.sqlite').exists())
        with self.connector(): asyncio.run(run())

    def test_changed_credentials_refuse_before_owner(self):
        app=self.app();self.credential.write_bytes(b'changed')
        async def run():
            with patch.object(launch,'create_native_owner') as owner:
                with self.assertRaisesRegex(ValueError,'native_launch_inputs_changed'):
                    async with app.router.lifespan_context(app): pass
                owner.assert_not_called()
            self.assertTrue(app.state.installed_lifecycle['closed'])
        asyncio.run(run())

    def test_cancel_acquisition_and_failed_close_retains_strong_owner(self):
        for fail in (False,True):
            with self.subTest(fail=fail):
                # Separate enrollment fixture for each case.
                fixture=NativeLaunchTests();fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                app=fixture.app();entered=threading.Event();release=threading.Event();closed=[]
                class Owner:
                    def close(self):
                        closed.append(True)
                        if fail: raise RuntimeError('close_unknown')
                owner=Owner()
                def acquire(**kwargs):
                    entered.set();self.assertTrue(release.wait(4));return owner
                async def run():
                    ctx=app.router.lifespan_context(app)
                    task=asyncio.create_task(ctx.__aenter__())
                    self.assertTrue(await asyncio.to_thread(entered.wait,4))
                    task.cancel()
                    await asyncio.sleep(.02)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError): await task
                    record=app.state.installed_lifecycle
                    self.assertIn(id(record),e._INSTALLED_LIFECYCLES)
                    release.set()
                    try:
                        await asyncio.wait_for(asyncio.shield(record['task']),4)
                    except RuntimeError:
                        self.assertTrue(fail)
                    self.assertIs(record['owner'],owner);self.assertEqual(closed,[True])
                    self.assertEqual(record['closed'],not fail)
                    self.assertEqual(id(record) in e._INSTALLED_LIFECYCLES,fail)
                    # Test-only fake failed obligation, no product resource.
                    e._INSTALLED_LIFECYCLES.pop(id(record),None)
                with patch.object(launch,'create_native_owner',side_effect=acquire): asyncio.run(run())

    def test_credential_policy_and_principal_scope_refusals_before_tls(self):
        app=self.app()
        kwargs=dict(record_file=str(self.record), configuration_file=str(self.root/'control'),
            templates_file=str(self.root/'templates'),owner_root=str(self.owner),
            staging=str(self.f.stage),credential_file=str(self.credential))
        requests=len(self.f.requests)
        for principal, scopes in [('bob',['read']),('alice',['production-write']),('alice',['read','admin'])]:
            self.write('credentials',canonical(dict(version=1,credentials=[dict(token='x'*64,principal=principal,scopes=scopes)])))
            with self.assertRaises(ValueError):launch.create_native_app(**kwargs)
        self.credential.chmod(0o644)
        with self.assertRaises(Exception):launch.create_native_app(**kwargs)
        self.assertEqual(len(self.f.requests),requests)

    def test_cli_success_uses_explicit_loopback_no_actual_listener(self):
        from types import SimpleNamespace
        app=SimpleNamespace(state=SimpleNamespace(installed_lifecycle=dict(closed=True,error=None)))
        class Server:
            started=True
            async def serve(self): pass
        args=['serve']
        for name in ('record-file','configuration-file','templates-file','owner-root','staging','credential-file'):
            args.extend(['--'+name,'/explicit'])
        with patch.object(launch,'create_native_app',return_value=app) as factory, patch.object(launch,'_installed_server',return_value=Server()) as server:
            launch.main(args+['--host','127.0.0.1','--port','8780'])
            config=server.call_args.args[0]
            self.assertEqual((config.host,config.port,config.access_log),('127.0.0.1',8780,False))
            self.assertEqual(factory.call_args.kwargs['record_file'],'/explicit')

    def test_cli_refuses_arguments_without_owner_or_listener(self):
        with patch.object(launch,'create_native_app') as factory:
            with self.assertRaises(SystemExit): launch.main(['serve'])
            args=['serve']
            for name in ('record-file','configuration-file','templates-file','owner-root','staging','credential-file'):
                args.extend(['--'+name,'/explicit'])
            with self.assertRaises(SystemExit):launch.main(args+['--host','0.0.0.0','--port','8000'])
            with self.assertRaises(SystemExit) as caught:launch.main(args+['--host','127.0.0.1','--port','0'])
            self.assertNotIn('/explicit',str(caught.exception));factory.assert_not_called()
