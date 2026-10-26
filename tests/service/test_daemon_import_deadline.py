"""Tiny streams and controlled clocks; no model runtime or Docker."""
import base64
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock
from service import daemon_helper as dh, daemon_client as dc
from service import installed_qwen_daemon as qd, installed_pocket_daemon as pd
from tests.service.test_qwen_daemon import QwenDaemonTests


class MainEntrypointTests(unittest.TestCase):
    def test_install_phase_budgets_after_volume_binding(self):
        # Inject Docker and helper responses, but execute main's actual phase
        # loop (the clock-only tests never reached its family selection).
        from service import install_policy as ip
        image='sha256:'+'a'*64
        config={'images':{'control':image,'worker':image},
                'budget':{'ram_bytes':1024,'pids':16}}
        inputs=[dict(role='config',path='install.json',source='config',bytes=1,sha256='b'*64),
                dict(role='credential',path='control.key',source='key',bytes=1)]
        request=dict(configuration=config,helper_image=image,inputs=inputs,
                     cpu=2,tmpfs=128,control_port=8000)
        for label,entrypoint,family in (('Qwen',dc.qwen_main,qd.FAMILY),
                                       ('Pocket',dc.pocket_main,pd.FAMILY),
                                       ('Kokoro',dc.main,None)):
            with self.subTest(family=label):
                docker=Mock(frozen={'endpoint':'injected'},end=220)
                volumes={}
                observed=[]
                def call(args):
                    if args[:2]==['image','inspect']:return image
                    if args[:2]==['volume','ls']:return ''
                    if args[:2]==['volume','create']:
                        name=args[-1]
                        volumes[name]=dict(Name=name,Driver='local',Options=None,Scope='local',
                            Mountpoint='/injected',CreatedAt='injected',Labels={})
                        return name
                    self.assertEqual(args,['injected-helper'])
                    return 'c'*64
                def exchange(identity,phase,nonce,configuration,binding,entries):
                    self.assertEqual(len(binding['volumes']),len(dc.dv.ROLES))
                    self.assertEqual(entries,inputs if phase=='install' else [])
                    observed.append((phase,docker.end))
                    return {'phase':phase}
                docker.call.side_effect=call
                docker.json.side_effect=lambda args:[volumes[args[-1]]]
                docker.exchange.side_effect=exchange
                def command(*args,**kwargs):
                    # Assert the deadline already exists before helper creation.
                    expected=700 if family is qd.FAMILY and kwargs['phase']=='install' else 220
                    self.assertEqual(docker.end,expected)
                    return ['injected-helper']
                with patch.object(dc,'Docker',return_value=docker) as constructor, \
                     patch.object(dc,'read',return_value=dc.encoded(request)), \
                     patch.object(dc,'exclusive') as receipt, \
                     patch.object(ip,'family_configuration',return_value=config), \
                     patch.object(dc.dv,'binding'),patch.object(dc.dv,'volume'), \
                     patch.object(dc.dv,'command',side_effect=command), \
                     patch('time.monotonic',return_value=100),patch('builtins.print'):
                    entrypoint(['install','--context','injected','--request','request','--handle','handle'])
                constructor.assert_called_once_with('injected',family=family)
                self.assertEqual(observed,[('install',700 if family is qd.FAMILY else 220),('init',220)])
                self.assertTrue(any(c.args[0]=='handle.binding' for c in receipt.call_args_list))

    def test_unknown_family_refused_before_docker_or_receipts(self):
        with patch.object(dc,'Docker') as docker,patch.object(dc,'exclusive') as receipt, \
             self.assertRaisesRegex(ValueError,'explicit_installed_family_required'):
            dc.main(['install','--handle','handle'],family=object())
        docker.assert_not_called();receipt.assert_not_called()


class DeadlineTests(unittest.TestCase):
    def test_handshake_converts_once_and_rejects_injected_budget(self):
        for family,phase in ((None,'install'),(pd.FAMILY,'install'),(qd.FAMILY,'init')):
            wire=SimpleNamespace(receive=Mock(return_value=('import-deadline',{'expires':1600})))
            with self.assertRaisesRegex(ValueError,'daemon_helper_begin'):dh.begin(wire,phase,family)
        for expires in (True, None, float('nan'), float('inf'), 1000, 1601):
            with patch('time.time',return_value=1000),self.assertRaises(ValueError):qd.import_deadline(expires)
        wire=SimpleNamespace(receive=Mock(side_effect=[('import-deadline',{'expires':1600}),('begin',{'inert':True})]))
        with patch('time.time',return_value=1010),patch('time.monotonic',return_value=20):
            request,end=dh.begin(wire,'install',qd.FAMILY)
        self.assertEqual(end,610);self.assertEqual(wire.deadline,end)
        self.assertEqual(request,{'inert':True})

    def test_receiver_rejects_late_response_without_renewing(self):
        h=object.__new__(dh.Helper);h.end=600;h.stable=Mock()
        h.wire=SimpleNamespace(send=Mock(),receive=Mock(return_value=('input',{})))
        with patch('time.monotonic',side_effect=[599,601]),self.assertRaisesRegex(ValueError,'daemon_helper_deadline'):
            h.rpc('input',{})
        self.assertEqual(h.end,600)

    def test_sender_wire_stall_uses_original_expiry(self):
        with patch.object(dc.os,'set_blocking'):
            wire=dc.Wire(Mock(),Mock(),'a'*32,'install',deadline=600)
        selector=Mock();selector.__enter__=Mock(return_value=selector);selector.__exit__=Mock(return_value=False)
        selector.select.return_value=[]
        with patch('time.monotonic',return_value=550),patch.object(dc.selectors,'DefaultSelector',return_value=selector),self.assertRaisesRegex(ValueError,'daemon_phase_timeout'):
            wire.receive()
        selector.select.assert_called_once_with(50)
        self.assertEqual(wire.deadline,600)


class StreamDeadlineTests(QwenDaemonTests):
    def test_stream_exceeds_old_100_seconds_and_completes_under_cap(self):
        # Tiny files still take several input RPCs, with a fixed expiry.
        s=self.session();count=len(s.inputs)
        original=s.action;now=[0.0];original_init=dh.Helper.__init__
        def action(name,payload):
            result=original(name,payload)
            if name=='input':now[0]+=150/count
            return result
        def initialize(h,*args,**kwargs):
            original_init(h,*args,**kwargs);h.end=600
        s.action=action
        with patch.object(dh.Helper,'__init__',initialize),patch('time.monotonic',side_effect=lambda:now[0]):
            self.assertEqual(s.run('install'),{'phase':'imported'})
        self.assertGreater(now[0],100);self.assertLess(now[0],600)
        self.assertEqual(s.helper.end,600)
        self.assertTrue((s.roots['authority']/'import-done.json').exists())

    def test_expiry_across_chunks_retains_intent_without_success(self):
        s=self.session();original=s.action;now=[0.0];original_init=dh.Helper.__init__
        def action(name,payload):
            result=original(name,payload)
            if name=='input':
                now[0]+=301
                result={'data':base64.b64encode(s.bytes[payload['index']][payload['offset']:payload['offset']+1]).decode()}
            return result
        def initialize(h,*args,**kwargs):original_init(h,*args,**kwargs);h.end=600
        s.action=action
        with patch.object(dh.Helper,'__init__',initialize),patch.object(dh,'INPUT_CHUNK',1),patch('time.monotonic',side_effect=lambda:now[0]),self.assertRaises(ValueError):s.run('install')
        self.assertTrue((s.roots['authority']/'import-intent.json').exists())
        self.assertFalse((s.roots['authority']/'import-done.json').exists())
