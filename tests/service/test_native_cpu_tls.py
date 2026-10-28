"""Real local TLS prerequisites, no model or installed qualification."""
import hashlib
from pathlib import Path
import socket
import ssl
import tempfile
import threading
import unittest

from service.native_cpu_install import create_tls,certificate_pin
from worker.native_cpu_protocol import tls_context


class CPUTLSTests(unittest.TestCase):
    def test_generated_private_credentials_mutually_authenticate_tls13(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();create_tls(root)
            for path in (root/'tls').iterdir():
                self.assertEqual(path.stat().st_mode&0o777,0o600)
            def context(role):
                return tls_context(server=role=='server',cafile=str(root/'tls/ca.pem'),
                    certfile=str(root/'tls'/(role+'.pem')),keyfile=str(root/'tls'/(role+'.key')))
            server,client=context('server'),context('client')
            observed=[];errors=[]
            with socket.socket() as listener:
                listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(2)
                def accept():
                    try:
                        raw,_=listener.accept();raw.settimeout(2)
                        with server.wrap_socket(raw,server_side=True) as connection:
                            observed.append((connection.version(),hashlib.sha256(connection.getpeercert(binary_form=True)).hexdigest()))
                            self.assertEqual(connection.recv(1),b'x');connection.sendall(b'y')
                    except BaseException as exc:errors.append(exc)
                thread=threading.Thread(target=accept);thread.start()
                try:
                    with socket.create_connection(listener.getsockname(),timeout=2) as raw:
                        with client.wrap_socket(raw,server_hostname='127.0.0.1') as connection:
                            self.assertEqual(connection.version(),'TLSv1.3')
                            self.assertEqual(hashlib.sha256(connection.getpeercert(binary_form=True)).hexdigest(),certificate_pin(root,'server'))
                            connection.sendall(b'x');self.assertEqual(connection.recv(1),b'y')
                finally:thread.join(3)
                self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
            self.assertEqual(observed,[('TLSv1.3',certificate_pin(root,'client'))])
            self.assertNotEqual(certificate_pin(root,'client'),certificate_pin(root,'server'))
            self.assertEqual(server.verify_mode,ssl.CERT_REQUIRED)


if __name__=='__main__':unittest.main()
