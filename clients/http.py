"""Shared bounded HTTP transport. No client-owned business state."""
import os
from urllib.parse import quote
import httpx
from contracts.validate import validate
from service.catalogue import Failure, IMPLEMENTED, OPERATIONS, VOICE_READS

class HTTP:
    def __init__(self, base_url=None, *, credential=None):
        self.credential = credential
        self.base_url = base_url or os.environ.get('SPEECH_URL', 'http://127.0.0.1:8780')

    def get(self, path, params=None):
        try:
            with httpx.Client(base_url=self.base_url, timeout=10, trust_env=False, follow_redirects=False) as client:
                response = client.get(path, params=params, headers={'Authorization': 'Bearer ' + self.credential} if self.credential else {})
                body = response.json()
        except (httpx.HTTPError, ValueError):
            raise ConnectionError('Skeleton HTTP transport unavailable.') from None
        if response.status_code != 200:
            validate('error', body)
            error = Failure(body['code'], body['message'], body['field'], body['action'])
            error.body = body
            raise error
        return body

    def call(self, operation, payload):
        if operation not in (*IMPLEMENTED, *VOICE_READS):
            raise Failure('not_found', 'Operation unavailable in the 0C skeleton.', 'operation')
        op = OPERATIONS[operation]
        validate(op['request'], payload)
        route = op['http']['route']
        query = dict(payload)
        for key in tuple(query):
            marker = '{' + key + '}'
            if marker in route:
                route = route.replace(marker, quote(str(query.pop(key)), safe=''))
        return validate(op['response'], self.get(route, query))
