"""Closed public GitHub asset redirect exception; no HTTP/model acquisition."""
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.request
import io
import tempfile
import ssl
from service.provisioning import _AllowedRedirects,download

ORIGINS=['https://github.com','https://release-assets.githubusercontent.com']


class CPUPublicDownloadTests(unittest.TestCase):
    def test_public_assets_use_pinned_certifi_ca_and_hostname_verification(self):
        import certifi
        from service.provisioning import _download_stream
        with tempfile.TemporaryDirectory() as temporary,patch('ssl.create_default_context',wraps=ssl.create_default_context) as context,patch('urllib.request.build_opener') as opener:
            opener.return_value.open.return_value=io.BytesIO(b'abc')
            target=Path(temporary)/'asset.part'
            _download_stream('https://github.com/example/release',target,3,ORIGINS,True)
            context.assert_called_once_with(cafile=certifi.where())
            handlers=opener.call_args.args
            https=next(h for h in handlers if type(h) is urllib.request.HTTPSHandler)
            self.assertTrue(https._context.check_hostname)
            self.assertEqual(https._context.verify_mode,ssl.CERT_REQUIRED)
            self.assertGreater(https._context.cert_store_stats()['x509_ca'],0)
            self.assertEqual(next(h for h in handlers if type(h) is urllib.request.ProxyHandler).proxies,{})
            self.assertEqual(target.read_bytes(),b'abc')

    def test_signed_query_only_on_explicit_release_asset_host(self):
        request=urllib.request.Request('https://github.com/example/release')
        url='https://release-assets.githubusercontent.com/example?sig=inert-fixture'
        for policy in (_AllowedRedirects(ORIGINS),_AllowedRedirects(['https://huggingface.co'])):
            with self.assertRaises(ValueError):policy.redirect_request(request,None,302,'Found',{},url)
        result=_AllowedRedirects(ORIGINS,True).redirect_request(request,None,302,'Found',{},url)
        self.assertEqual(result.full_url,url)
        for bad in ('http://release-assets.githubusercontent.com/example','https://github.com/example?sig=inert',
                'https://elsewhere.invalid/example?sig=inert','https://name:password@release-assets.githubusercontent.com/example',
                'https://release-assets.githubusercontent.com/example#fragment'):
            with self.subTest(url=bad),self.assertRaises(ValueError):_AllowedRedirects(ORIGINS,True).redirect_request(request,None,302,'Found',{},bad)

    def test_transfer_policy_cannot_widen_huggingface_or_use_unbounded_size(self):
        with patch('service.provisioning.multiprocessing.get_context') as process:
            for url,size,origins in (('https://huggingface.co/model',10,ORIGINS),('https://github.com/model',None,ORIGINS),
                    ('https://github.com/model',10,['https://github.com','https://huggingface.co'])):
                with self.assertRaises(ValueError):download(url,Path('/must-not-write'),size,allowed_origins=origins,public_github_assets=True)
            process.assert_not_called()


if __name__=='__main__':unittest.main()
