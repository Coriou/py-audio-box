"""Inert framing/identity tests: fake TLS only, no listener or network."""
import base64
import copy
import hashlib
import struct
import time
import unittest

from worker import native_protocol as p
from worker.transport import Refused, Uncertain, canonical


class TLS:
    def __init__(self, raw=b''):
        self.raw = bytearray(raw)
        self.sent = b''
        self.cert = b'fake enrolled cert'
        self.protocol = 'TLSv1.3'
    def version(self): return self.protocol
    def getpeercert(self, binary_form): return self.cert
    def settimeout(self, seconds): self.timeout = seconds
    def recv(self, size):
        data = bytes(self.raw[:min(3, size)])
        del self.raw[:len(data)]
        return data
    def sendall(self, data): self.sent += data


class NativeProtocolTests(unittest.TestCase):
    def setUp(self):
        self.request = dict(version=p.VERSION, deployment_id='a'*32,
            configuration_sha256='b'*64, lane='c'*32, boot='d'*32,
            request_id='e'*32, runtime_id='c'*32+'/'+'d'*32+'/'+'f'*32,
            binding='1'*64, op='start', body={})
        self.pin = hashlib.sha256(b'fake enrolled cert').hexdigest()

    def test_closed_operations_and_allocation_tuple(self):
        for op in ('start', 'observe', 'stop', 'collect', 'ack'):
            self.assertEqual(p.request(dict(self.request, op=op))['op'], op)
        p.request(dict(self.request, op='hello', lane=None, boot=None, runtime_id=None, binding=None))
        p.request(dict(self.request, op='prepare', runtime_id=None, body={'snapshot':[{}, {}, {}, {}]}))
        for mutation in ({'op':'exec'}, {'extra':1}, {'binding':None},
                         {'runtime_id':'wrong'}, {'boot':'E'*32}, {'body':{'path':'/tmp/x'}}):
            with self.subTest(mutation=mutation), self.assertRaises(Refused):
                p.request(dict(self.request, **mutation))

    def test_chunks_canonical_and_bounded(self):
        body = dict(sha256='2'*64, bytes=4, offset=0, data=base64.b64encode(b'1234').decode())
        p.request(dict(self.request, op='put_input', body=body))
        for mutation in ({'bytes':True}, {'offset':True}, {'offset':1},
                         {'data':'MTIzNA==\n'}, {'data':'Zh=='},
                         {'data':base64.b64encode(b'x'*32769).decode()}, {'path':'x'}):
            with self.subTest(mutation=mutation), self.assertRaises(Refused):
                p.request(dict(self.request, op='put_input', body=dict(body, **mutation)))
        with self.assertRaises(Refused):
            p.request(dict(self.request, op='prepare', runtime_id=None,
                           body={'snapshot':['x'*48000, {}, {}, {}]}))

    def test_tls_pin_and_canonical_frames(self):
        connection = TLS()
        p.send(connection, self.request, time.monotonic()+2, peer_sha256=self.pin)
        self.assertEqual(p.receive(TLS(connection.sent), time.monotonic()+2,
                                  peer_sha256=self.pin), self.request)
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{ "a": 1 }', b'\xff'):
            with self.subTest(raw=raw), self.assertRaises(Refused):
                p.receive(TLS(struct.pack('!I',len(raw))+raw), time.monotonic()+2, peer_sha256=self.pin)
        for size in (0, 65537):
            with self.assertRaises(Refused):
                p.receive(TLS(struct.pack('!I',size)), time.monotonic()+2, peer_sha256=self.pin)
        with self.assertRaises(Uncertain):
            p.receive(TLS(struct.pack('!I',10)+b'{}'), time.monotonic()+2, peer_sha256=self.pin)
        for cert, protocol in ((b'wrong', 'TLSv1.3'), (b'', 'TLSv1.3'), (b'fake enrolled cert', 'TLSv1.2')):
            connection = TLS(); connection.cert = cert; connection.protocol = protocol
            with self.assertRaises(Refused):
                p.send(connection, {}, time.monotonic()+2, peer_sha256=self.pin)

    def test_explicit_bind(self):
        self.assertEqual(p.numeric_bind('192.168.65.1', 1234), ('192.168.65.1',1234))
        for host in ('0.0.0.0', '::', '127.0.0.1', '::1', 'localhost', '224.0.0.1'):
            with self.subTest(host=host), self.assertRaises(Refused): p.numeric_bind(host, 1234)
        self.assertEqual(p.numeric_bind('127.0.0.1', 1234, proven_host_network=True), ('127.0.0.1',1234))
        with self.assertRaises(Refused): p.numeric_bind('192.168.65.1', True)

    def test_receipt_requires_trusted_full_binding_and_owned_wait_shape(self):
        expected = {k:self.request[k] for k in ('runtime_id','binding','deployment_id','configuration_sha256','lane','boot')}
        expected['original_deadline'] = '2026-10-25T12:00:00+00:00'
        receipt = dict(expected, version='native-receipt-v1', reaped=True,
            timely=True, receiver_closed=True, reap_method='darwin-owned-exec-v1',
            processes=[dict(role='model', spawn_token='a'*32, pid=10, pgid=10,
                            start_identity=None, exit_code=0)])
        self.assertIs(p.receipt(receipt, expected=expected), receipt)
        for mutation in ({'engine_uid':1000}, {'reap_method':'pid1-waitpid-echild'},
                         {'binding':'3'*64}, {'receiver_closed':False},
                         {'reaped':1}, {'timely':1}, {'processes':[]}):
            with self.subTest(mutation=mutation), self.assertRaises(Refused):
                p.receipt(dict(receipt, **mutation), expected=expected)
        for mutation in ({'pid':True}, {'exit_code':True}, {'spawn_token':'bad'}, {'role':'helper'}):
            changed=copy.deepcopy(receipt); changed['processes'][0].update(mutation)
            with self.assertRaises(Refused): p.receipt(changed, expected=expected)
        p.receipt(dict(receipt, timely=False, processes=[]), expected=expected)


if __name__ == '__main__': unittest.main()
