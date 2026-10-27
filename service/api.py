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
    return JSONResponse(problem('not_found', 'Route or method unavailable.'), status_code=404)

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

@app.get('/v1/status')
@app.get('/skeleton', deprecated=True)
def summary(query: Annotated[EmptyQuery, Query()] = EmptyQuery()):
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


def create_voice_app(dispatcher, *, credentials, upload_deadline=15.0, installed=None, native_installed=False):
    """Borrow an initialized Dispatcher. Caller retains sole close responsibility.

    This factory starts no owner, worker, listener or persistent enrollment.
    """
    from service.domain import voice_read, asset_call
    from service.catalogue import VOICE_READS, schema, ASSET_OPERATIONS, PRODUCTION_OPERATIONS
    if not callable(getattr(credentials, 'resolve', None)):
        raise ValueError('credential_resolver_required')
    if installed is not None:
        from service import install_policy as ip
        from service.entrypoint import InstalledActivation
        from service import installed_export as ex
        selected = (type(dispatcher.installed_activation) is InstalledActivation and
                    dispatcher.installed_activation.family is ex.FAMILY)
        installed = ex.configuration(installed) if selected else ip.configuration(installed)
        if (not dispatcher.installed_checked or
                dispatcher.installed_activation.configuration != installed):
            raise ValueError('checked_installed_activation_required')
    if native_installed:
        from service.native_activation import NativeActivation
        if installed is not None or type(dispatcher.installed_activation) is not NativeActivation or not dispatcher.installed_checked:
            raise ValueError('checked_native_activation_required')
    production = FastAPI(title='Authenticated installed speech control' if installed is not None or native_installed else 'Authenticated synthetic production control', docs_url=None,
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
    def discovery(request: Request, query: Annotated[EmptyQuery, Query()]):
        body = capabilities()
        configured = dispatcher.call('configured_capabilities', request.state.principal)
        if configured:
            body['profiles'] = [configured['profile']]
            body['engines'] = [configured['engine']]
        if installed is not None:
            import copy
            from datetime import datetime, timezone
            body['profiles'] = copy.deepcopy(installed['profiles'])
            body['engines'] = sorted({c['identity']['engine'] for p in installed['profiles'] for c in p['compatibility']})
            body['checked_at'] = datetime.now(timezone.utc).isoformat()
            for profile in body['profiles']:
                profile['next_action'] = configured['next_action'] if configured and configured['engine']=='pocket-tts' else 'Read /v1/status configured_presets; explicitly call voice_preset_register, then speech_generate. Clone preparation is unavailable.'
            if not dispatcher.call('status')['persistent_activation']:
                for profile in body['profiles']: profile['readiness'] = 'not_ready'
        selected_profiles = dispatcher.call('configured_profiles', request.state.principal)
        if selected_profiles:
            body['profiles'] = selected_profiles
            body['engines'] = sorted({c['identity']['engine'] for p in selected_profiles for c in p['compatibility']})
        body['operations'] = list(dict.fromkeys(('discover', *VOICE_READS, *ASSET_OPERATIONS, *PRODUCTION_OPERATIONS)))
        if configured:
            body['operations'] = [op for op in body['operations'] if op not in ('voice_preset_register','voice_prepare','voice_results','voice_save','speech_generate') or op in configured['operations']]
        if installed is not None and not (configured and configured['engine']=='pocket-tts'):
            body['operations'] = [op for op in body['operations'] if op not in ('voice_prepare', 'voice_results', 'voice_save')]
        if not dispatcher.call('audio_export_available',request.state.principal): body['operations'] = [op for op in body['operations'] if op != 'audio_export']
        if not dispatcher.call('status')['retry_available']: body['operations'] = [op for op in body['operations'] if op != 'job_retry']
        return body

    @production.get('/v1/status')
    @production.get('/skeleton', deprecated=True)
    def production_status(request: Request, query: Annotated[EmptyQuery, Query()]):
        if installed is not None:
            from service.persistent_init import digest
            status = dispatcher.call('status')
            configured = dispatcher.call('configured_capabilities', request.state.principal)
            clone = bool(configured and configured['engine']=='pocket-tts')
            profile = configured['profile'] if clone else None
            qualified = bool(profile and profile['support']=='qualified' and profile['readiness']=='ready' and
                             profile['verification']=='measured' and profile['qualification_receipt'] and
                             'synthetic' not in profile['qualification_receipt'].lower()) if clone else True
            return dict(synthetic=False, ready_for_speech=status['persistent_activation'] and qualified,
                        configuration_digest=digest(installed),
                        profiles=installed['profiles'], store='installed-owner', status=status,
                        configured_presets=dispatcher.call('configured_presets', request.state.principal),
                        next_action=configured['next_action'] if clone else 'voice_preset_register → voices_list/voice_version_get → speech_generate',
                        clone_prepare=clone, installed=True, configured_capabilities=configured,
                        activation_ready=status['persistent_activation'],
                        qualification='operator-declared configured profile; installation and registration do not qualify execution')
        configured = dispatcher.call('configured_capabilities', request.state.principal)
        if native_installed:
            status = dispatcher.call('status')
            return dict(synthetic=False, ready_for_speech=False, installed=True,
                        activation_ready=status['persistent_activation'], status=status,
                        configured_capabilities=configured, store='installed-native-owner',
                        configured_presets=dispatcher.call('configured_presets', request.state.principal),
                        qualification='UNAVAILABLE / UNQUALIFIED; source activation does not qualify speech')
        if configured:
            return dict(synthetic=False, ready_for_speech=False, installed=False,
                        configured_capabilities=configured, store='borrowed-owner',
                        qualification='See configured profile; source configuration does not qualify installed execution.')
        return dict(synthetic=True, ready_for_speech=False, engine_qualification='none',
                    context_configuration='operator-required', store='borrowed-owner')

    @production.get('/skeleton/operations/{operation_id}')
    def voice_detail(operation_id: str, request: Request):
        if operation_id not in ('discover', *VOICE_READS, *ASSET_OPERATIONS, *PRODUCTION_OPERATIONS):
            raise Failure('not_found', 'Operation unavailable.', 'operation_id')
        op = OPERATIONS[operation_id]
        availability = 'implemented' if op['http']['method'] == 'POST' else 'implemented_read'
        configured = dispatcher.call('configured_capabilities', request.state.principal)
        if operation_id == 'job_retry' and not dispatcher.call('status')['retry_available']: availability = 'unavailable_lifecycle'
        if configured and operation_id in ('voice_preset_register','voice_prepare','voice_results','voice_save','speech_generate') and operation_id not in configured['operations']:
            availability = 'unavailable_configured_profile'
        if (installed is not None and operation_id in ('voice_prepare', 'voice_results', 'voice_save')
                and not (configured and operation_id in configured['operations'])):
            availability = 'unavailable_configured_profile'
        if operation_id=='audio_export': availability=dispatcher.call('audio_export_availability',request.state.principal)
        return dict(fixture=False, availability=availability, operation=op,
                    request_schema=schema(op['request']), response_schema=schema(op['response']))

    @production.get('/v1/voices')
    def listing(request: Request, query: Annotated[VoicesQuery, Query()]):
        return voice_read(dispatcher, request.state.principal, 'voices_list',
                          dict(limit=query.limit, cursor=query.cursor or ''))

    @production.get('/v1/voices/{voice_id}')
    def metadata(voice_id: str, request: Request, query: Annotated[EmptyQuery, Query()]):
        return voice_read(dispatcher, request.state.principal, 'voice_get', dict(voice_id=voice_id))

    @production.get('/v1/voice-names/{name}')
    def resolve(name: str, request: Request, query: Annotated[EmptyQuery, Query()]):
        return voice_read(dispatcher, request.state.principal, 'voice_resolve', dict(name=name))

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
                            if length is not None and total > length: raise Failure('invalid_request', 'Content-Length mismatch.')
                            # ASGI message size is server-controlled. The total cap
                            # bounds our retained body, not upstream server allocations.
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
    from service.domain import production_call
    def control_read_endpoint(operation):
        async def endpoint(request: Request):
            from contracts.validate import validate, ContractError
            from jsonschema.exceptions import ValidationError
            from service.job_control import wait_job
            payload = dict(request.path_params)
            if len(request.query_params.multi_items()) != len(set(request.query_params.keys())):
                raise Failure('invalid_request', 'Duplicate query fields.')
            if set(payload).intersection(request.query_params):
                raise Failure('invalid_request', 'Path fields cannot be repeated in the query.')
            payload.update(request.query_params)
            try:
                for name in ('limit', 'wait_seconds'):
                    if name in payload:
                        value = payload[name]
                        if not value.isascii() or not value.isdecimal(): raise ValueError()
                        payload[name] = int(value)
                validate(OPERATIONS[operation]['request'],payload)
            except (ValueError, ValidationError, ContractError):
                raise Failure('invalid_request', 'Use the named bounded query contract.') from None
            if operation == 'job_wait':
                return await wait_job(dispatcher,request.state.principal,payload,request.is_disconnected)
            return await asyncio.to_thread(production_call,dispatcher,request.state.principal,operation,payload)
        return endpoint

    def read_endpoint(operation):
        def endpoint(request: Request, query: Annotated[EmptyQuery, Query()]):
            return production_call(dispatcher, request.state.principal, operation, dict(request.path_params))
        return endpoint

    def mutation_endpoint(operation):
        async def endpoint(request: Request, query: Annotated[EmptyQuery, Query()]):
            principal = credentials.resolve(request.headers.get('authorization'), required_scope='production-write')
            if request.headers.get('content-type', '').split(';')[0].strip().lower() != 'application/json':
                raise Failure('invalid_request', 'Use application/json.')
            keys = request.headers.getlist('idempotency-key')
            if len(keys) != 1:
                raise Failure('invalid_request', 'Supply one Idempotency-Key.')
            raw = bytearray()
            try:
                async with asyncio.timeout(upload_deadline):
                    async for chunk in request.stream():
                        if len(raw)+len(chunk) > 16384: raise Failure('resource_limit', 'Request exceeds 16 KiB.')
                        raw.extend(chunk)
                def pairs(items):
                    result = {}
                    for name, value in items:
                        if name in result: raise ValueError('duplicate_field')
                        result[name] = value
                    return result
                payload = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            except (ValueError, RecursionError, TimeoutError, ClientDisconnect):
                raise Failure('invalid_request', 'Invalid or incomplete JSON request.') from None
            return await asyncio.to_thread(production_call, dispatcher, principal, operation, payload, keys[0])
        return endpoint

    for operation in PRODUCTION_OPERATIONS:
        op = OPERATIONS[operation]
        production.add_api_route(op['http']['route'],
            (control_read_endpoint(operation) if operation in ('jobs_list','job_events','job_wait','profile_get') else read_endpoint(operation)) if op['http']['method'] == 'GET' else mutation_endpoint(operation),
            methods=[op['http']['method']])
    production.add_api_route('/skeleton/profiles/{profile_id}',control_read_endpoint('profile_get'),methods=['GET'])
    return production
