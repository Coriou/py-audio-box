import os
import socket
import subprocess
import sys
import time
import httpx
import pytest

@pytest.fixture(scope='session')
def server_url():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    process = subprocess.Popen([sys.executable,'-m','uvicorn','service.api:app','--host','127.0.0.1','--port',str(port),'--no-access-log'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if process.poll() is not None: raise RuntimeError('Fixture HTTP process exited')
            try:
                if httpx.get(url+'/skeleton',trust_env=False).status_code == 200: break
            except httpx.HTTPError: pass
            time.sleep(.05)
        else: raise RuntimeError('Fixture HTTP startup timed out')
        yield url
    finally:
        process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)

@pytest.fixture
def configured_url(server_url, monkeypatch):
    monkeypatch.setenv('SPEECH_URL',server_url)
    return server_url
