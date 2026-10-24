"""Default 0C fixture host and an explicit borrowed-owner voice read factory."""
import os
from pathlib import Path
from typing import Annotated
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException
from service.catalogue import (Failure, IMPLEMENTED, OPERATIONS, FIXTURES,
                               capabilities, operation_detail, problem)
from service.domain import read
import json

app = FastAPI(title='0C synthetic evaluation skeleton', docs_url=None, redoc_url=None)

class EmptyQuery(BaseModel):
    model_config = ConfigDict(extra='forbid')

class VoicesQuery(EmptyQuery):
    limit: int = Field(default=20, ge=1, le=20)
    cursor: str | None = None

class ContentQuery(EmptyQuery):
    start: int = Field(ge=0, le=9007199254740991)
    end_exclusive: int = Field(ge=0, le=9007199254740991)

@app.exception_handler(Failure)
async def failure_handler(request, exc):
    return JSONResponse(exc.body, status_code=exc.status)

@app.exception_handler(RequestValidationError)
async def invalid_handler(request, exc):
    return JSONResponse(problem('invalid_request', 'Invalid request fields.', 'request'), status_code=422)

@app.exception_handler(HTTPException)
async def http_handler(request, exc):
    return JSONResponse(problem('not_found', 'Route or method unavailable in the 0C skeleton.'), status_code=404)

@app.middleware('http')
async def fixture_header(request: Request, call_next):
    response = await call_next(request)
    response.headers['X-Speech-Fixture'] = 'synthetic-0C-no-service-store'
    return response

@app.get('/v1/capabilities')
def discover(query: Annotated[EmptyQuery, Query()]):
    return read('discover', {})

@app.get('/v1/voices')
def voices(query: Annotated[VoicesQuery, Query()]):
    return read('voices_list', dict(limit=query.limit, cursor=query.cursor or ''))

@app.get('/v1/assets/{asset_id}')
def asset(asset_id: str, query: Annotated[EmptyQuery, Query()]):
    return read('asset_get', {'asset_id': asset_id})

@app.get('/v1/assets/{asset_id}/content')
def content(asset_id: str, query: Annotated[ContentQuery, Query()]):
    return read('asset_content', {'asset_id': asset_id, **query.model_dump()})

@app.get('/skeleton')
def summary():
    return {'fixture': True, 'contract_version': '1.3.0', 'ready_for_speech': False,
        'implemented': list(IMPLEMENTED), 'unavailable_operations': len(OPERATIONS)-len(IMPLEMENTED),
        'details': '/skeleton/operations/{operation_id}', 'profiles': '/skeleton/profiles/{profile_id}',
        'ui': '/evaluation/', 'store': 'none',
        'source_types_note': 'Canonical source_types names intended sources; acquisition is unavailable.',
        'byte_transfer': 'Explicit asset_content only; direct MCP 3072 decoded bytes; larger bytes via CLI file.',
        'worker': 'Optional networkless UDS boundary probe only; no job execution.'}

@app.get('/skeleton/operations/{operation_id}')
def detail(operation_id: str):
    return operation_detail(operation_id)

@app.get('/skeleton/profiles/{profile_id}')
def profile(profile_id: str):
    profiles = json.loads((FIXTURES / 'capabilities.json').read_text())['profiles']
    for item in profiles:
        if item['id'] == profile_id: return item
    raise Failure('not_found', 'Unknown unqualified fixture profile.', 'profile_id')

@app.get('/', response_class=HTMLResponse)
def index():
    return '''<!doctype html><html lang="en"><title>0C synthetic evaluation</title>
    <h1>Synthetic evaluation only</h1><p>No speech generation, credentials, service persistence or qualified ML profiles.</p>
    <p>The editor saves locally in this browser. It is not connected to the service API.</p>
    <a href="/evaluation/">Open unchanged 0B editor evaluation</a> · <a href="/skeleton">Read skeleton status</a></html>'''

# Compiled 0B fixture is packaged under an explicitly synthetic evaluation path.
ui = Path(os.environ.get('SPEECH_EVALUATION_UI', '/opt/speech/evaluation'))
if ui.is_dir():
    app.mount('/evaluation', StaticFiles(directory=ui, html=True), name='synthetic-evaluation')


def create_voice_app(dispatcher, *, credentials, upload_deadline=15.0):
    """Borrow an initialized Dispatcher. Caller retains sole close responsibility.

    This factory starts no owner, worker, listener or persistent enrollment.
    """
    from service.domain import voice_read, asset_call
    from service.catalogue import VOICE_READS, schema, ASSET_OPERATIONS
    if not callable(getattr(credentials, 'resolve', None)):
        raise ValueError('credential_resolver_required')
    production = FastAPI(title='Authenticated voice catalogue reads', docs_url=None,
                         redoc_url=None, openapi_url=None)
    production.add_exception_handler(Failure, failure_handler)
    production.add_exception_handler(RequestValidationError, invalid_handler)
    production.add_exception_handler(HTTPException, http_handler)

    import asyncio
    import io
    import threading
    from starlette.requests import ClientDisconnect
    from service.ingest import MAX_BYTES
    upload_slot = threading.BoundedSemaphore(1)
    if not 0 < upload_deadline <= 60: raise ValueError('invalid_upload_deadline')

    @production.middleware('http')
    async def authenticate(request, call_next):
        try:
            request.state.principal = credentials.resolve(request.headers.get('authorization'))
        except Failure as exc:
            return JSONResponse(exc.body, status_code=exc.status)
        return await call_next(request)

    @production.get('/v1/capabilities')
    def discovery(query: Annotated[EmptyQuery, Query()]):
        body = capabilities()
        body['operations'] = ['discover', *VOICE_READS, *ASSET_OPERATIONS]
        return body

    @production.get('/skeleton/operations/{operation_id}')
    def voice_detail(operation_id: str):
        if operation_id not in ('discover', *VOICE_READS, *ASSET_OPERATIONS):
            raise Failure('not_found', 'Operation unavailable.', 'operation_id')
        op = OPERATIONS[operation_id]
        return dict(fixture=False, availability='implemented' if operation_id == 'asset_upload' else 'implemented_read', operation=op,
                    request_schema=schema(op['request']), response_schema=schema(op['response']))

    @production.get('/v1/voices')
    def listing(request: Request, query: Annotated[VoicesQuery, Query()]):
        return voice_read(dispatcher, request.state.principal, 'voices_list',
                          dict(limit=query.limit, cursor=query.cursor or ''))

    @production.get('/v1/voices/{voice_id}')
    def metadata(voice_id: str, request: Request, query: Annotated[EmptyQuery, Query()]):
        return voice_read(dispatcher, request.state.principal, 'voice_get', dict(voice_id=voice_id))

    @production.get('/v1/voice-versions/{voice_version_id}')
    def version(voice_version_id: str, request: Request, query: Annotated[EmptyQuery, Query()]):
        return voice_read(dispatcher, request.state.principal, 'voice_version_get',
                          dict(voice_version_id=voice_version_id))
    @production.get('/v1/voices/{voice_id}/versions')
    def versions(voice_id: str, request: Request, query: Annotated[VoicesQuery, Query()]):
        return voice_read(dispatcher, request.state.principal, 'voice_versions_list',
                          dict(voice_id=voice_id, limit=query.limit, cursor=query.cursor or ''))

    @production.post('/v1/assets')
    async def upload(request: Request, query: Annotated[EmptyQuery, Query()]):
        principal = credentials.resolve(request.headers.get('authorization'), required_scope='production-write')
        if request.headers.get('content-type', '').lower() != 'audio/wav':
            raise Failure('invalid_request', 'Use raw audio/wav bytes.')
        if len(request.headers.getlist('content-length')) > 1:
            raise Failure('invalid_request', 'Multiple Content-Length headers are unavailable.')
        length = request.headers.get('content-length')
        if length is not None:
            if not length.isascii() or not length.isdecimal() or len(length)>10:
                raise Failure('invalid_request', 'Invalid Content-Length.')
            length = int(length)
            if length > MAX_BYTES: raise Failure('resource_limit', 'Upload exceeds 24 MiB.')
        if not upload_slot.acquire(blocking=False):
            raise Failure('resource_limit', 'Upload admission is occupied.')
        release = True
        try:
            with io.BytesIO() as buffer:
                total = 0
                try:
                    async with asyncio.timeout(upload_deadline):
                        async for chunk in request.stream():
                            total += len(chunk)
                            if total > MAX_BYTES: raise Failure('resource_limit', 'Upload exceeds 24 MiB.')
                            if len(chunk)>65536: raise Failure('resource_limit', 'Transport chunks must be at most 65536 bytes.')
                            if length is not None and total > length: raise Failure('invalid_request', 'Content-Length mismatch.')
                            buffer.write(chunk)
                except (TimeoutError, ClientDisconnect):
                    raise Failure('invalid_request', 'Upload incomplete or deadline exceeded.') from None
                if total == 0 or (length is not None and total != length):
                    raise Failure('invalid_request', 'Content-Length mismatch or empty body.')
                # BytesIO.getvalue produces immutable bytes; network receive has ended.
                release = False  # Cancellation/unknown owner outcome retains admission.
                try:
                    result = await asyncio.to_thread(asset_call, dispatcher, principal, 'asset_upload', buffer.getvalue())
                    release = True
                    return result
                except Failure as exc:
                    release = exc.body['code'] != 'not_ready'
                    raise
        finally:
            if release: upload_slot.release()

    @production.get('/v1/assets/{asset_id}')
    def asset_metadata(asset_id: str, request: Request, query: Annotated[EmptyQuery, Query()]):
        return asset_call(dispatcher, request.state.principal, 'asset_get', dict(asset_id=asset_id))

    @production.get('/v1/assets/{asset_id}/content')
    def asset_bytes(asset_id: str, request: Request, query: Annotated[ContentQuery, Query()]):
        return asset_call(dispatcher, request.state.principal, 'asset_content', dict(asset_id=asset_id, **query.model_dump()))
    return production
