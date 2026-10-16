"""FastAPI 0C fixture host. No authentication, mutation, database or ML runtime."""
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
    return read('voices_list', query.model_dump(exclude_none=True))

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
