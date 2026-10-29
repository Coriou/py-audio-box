"""MCP v1 maintenance SDK, stdio; named reads call HTTP, never a command bus."""
import asyncio
import copy
import os
from collections import Counter
from urllib.parse import quote
from jsonschema.exceptions import ValidationError
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult
from contracts.validate import ContractError, validate
from clients.http import HTTP, network_os_error
from clients.config import ConfigurationFailure, InstallationFailure
from service.catalogue import (DIRECT_BYTES, Failure, IMPLEMENTED, VOICE_READS, PRODUCTION_OPERATIONS, OPERATIONS, compact, problem, schema)

server = Server('py-audio-box')


def inline_single_use_definitions(value):
    """Compact only self-contained, singly referenced local MCP definitions."""
    # URI and anchor scope can change what a local reference means after moving it.
    scope_keywords = {'$id', '$anchor', '$dynamicRef', '$dynamicAnchor',
                      '$recursiveRef', '$recursiveAnchor', '$schema', '$vocabulary'}
    def scoped(node):
        if isinstance(node, dict):
            return bool(scope_keywords.intersection(node)) or any(scoped(child) for child in node.values())
        return isinstance(node, list) and any(scoped(child) for child in node)
    if scoped(value):
        return copy.deepcopy(value)
    definitions = value.get('$defs', {})
    references = Counter()
    plain_references = Counter()

    def walk(node):
        if isinstance(node, dict):
            if '$ref' in node:
                references[node['$ref']] += 1
                if len(node) == 1:
                    plain_references[node['$ref']] += 1
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    replacements = {}
    for name, definition in definitions.items():
        ref = '#/$defs/' + name
        if references[ref] != 1 or plain_references[ref] != 1:
            continue
        nested = Counter()
        def refs(node):
            if isinstance(node, dict):
                if '$ref' in node:
                    nested[node['$ref']] += 1
                for child in node.values(): refs(child)
            elif isinstance(node, list):
                for child in node: refs(child)
        refs(definition)
        if not nested:
            replacements[ref] = definition

    def replace(node):
        if isinstance(node, dict):
            # A sibling of $ref can alter Draft 2020-12 semantics; leave it intact.
            if len(node) == 1 and node.get('$ref') in replacements:
                return copy.deepcopy(replacements[node['$ref']])
            return {key: replace(child) for key, child in node.items()}
        if isinstance(node, list):
            return [replace(child) for child in node]
        return node

    compacted = replace(value)
    if replacements:
        compacted['$defs'] = {name: definition for name, definition in compacted['$defs'].items()
                              if '#/$defs/' + name not in replacements}
        if not compacted['$defs']:
            del compacted['$defs']
    return compacted


def input_schema(operation):
    value = inline_single_use_definitions(schema(OPERATIONS[operation]['request']))
    if OPERATIONS[operation]['idempotency']:
        value['properties']['idempotency_key'] = {'type':'string','pattern':'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'}
        value['required'].append('idempotency_key')
    return value


def tools():
    items = [Tool(name=OPERATIONS[op]['mcp'],
        description=('Opt-in bytes, max 3072 decoded. Larger downloads: python -m clients.cli asset content ID --output FILE.' if op == 'asset_content' else 'Register a speech_status preset for generation only; no cloning or qualification.' if op == 'voice_preset_register' else None),
        inputSchema=input_schema(op)) for op in dict.fromkeys((*IMPLEMENTED, *VOICE_READS, *PRODUCTION_OPERATIONS))]
    if os.environ.get('SPEECH_UPLOAD_ROOT'):
        items.append(Tool(name='speech_asset_upload_file', description='Upload a client-host WAV from SPEECH_UPLOAD_ROOT. Operator-authorized root; relative no-follow paths only. Creates an asset, not a cloned voice. No inline audio.', inputSchema={'type':'object','properties':{'path':{'type':'string','minLength':1,'maxLength':4096}},'required':['path'],'additionalProperties':False}))
    items.append(Tool(name='speech_enroll', description='Save an uploaded trusted WAV as a named voice. Bounded wait returns job and asset handles on timeout. Resume with confirmed asset_id; bare WAV uploads create a new asset.', inputSchema={
        'type':'object','properties':{'name':{'type':'string','minLength':1,'maxLength':128},'asset_id':{'type':'string','minLength':1,'maxLength':128},
            'reference_text':{'type':'string','minLength':1,'maxLength':10000},'audition_text':{'type':'string','minLength':1,'maxLength':10000},
            'job_id':{'type':'string','minLength':1,'maxLength':128},'wait_seconds':{'type':'integer','minimum':0,'maximum':600}},
        'required':['name','asset_id','reference_text','audition_text'],'additionalProperties':False}))
    items.append(Tool(name='speech_speak', description='Generate from a saved named voice. Returns artifact and asset handles; bounded wait, replayable key, no inline audio.', inputSchema={
        'type':'object','properties':{'voice':{'type':'string','minLength':1,'maxLength':128},'text':{'type':'string','minLength':1,'maxLength':10000},
            'job_id':{'type':'string','minLength':1,'maxLength':128},'voice_version_id':{'type':'string','minLength':1,'maxLength':128},
            'wait_seconds':{'type':'integer','minimum':0,'maximum':600}},
        'required':['voice','text'],'additionalProperties':False}))
    items += [Tool(name='speech_status', description='Configured service status, preset selectors and next action; no discovery-time model loading.',
                   inputSchema={'type':'object','properties':{},'additionalProperties':False}),
              Tool(name='speech_operation_detail', description='Read one canonical operation schema and fixture availability; never executes it.',
                   inputSchema={'type':'object','properties':{'operation_id':{'type':'string','maxLength':128}},
                                'required':['operation_id'],'additionalProperties':False})]
    return items

@server.list_tools()
async def list_tools():
    return tools()


def result(body, error=False):
    # Both copies counted in measurement; hosts may add further copies.
    return CallToolResult(content=[TextContent(type='text', text=compact(body))],
                          structuredContent=body, isError=error)

@server.call_tool(validate_input=False)
async def call_tool(name, arguments):
    try:
        if not isinstance(arguments, dict):
            raise Failure('invalid_request', 'Expected named request fields.', 'request')
        from clients.config import resolve
        url, credential = resolve()
        transport = HTTP(url, credential=credential)
        if name in ('speech_enroll','speech_speak'):
            from jsonschema import Draft202012Validator
            tool = next(t for t in tools() if t.name == name)
            Draft202012Validator(tool.inputSchema).validate(arguments)
            from clients.workflow import enroll, speak
            if name == 'speech_enroll':
                body = await asyncio.to_thread(enroll, transport, name=arguments['name'],
                    asset_id=arguments['asset_id'], reference_text=arguments['reference_text'],
                    audition_text=arguments['audition_text'], job_id=arguments.get('job_id'),
                    seconds=arguments.get('wait_seconds',120))
            else:
                body = await asyncio.to_thread(speak, transport, voice=arguments['voice'],
                    text=arguments['text'], job_id=arguments.get('job_id'),
                    voice_version_id=arguments.get('voice_version_id'),
                    seconds=arguments.get('wait_seconds',120))
        elif name == 'speech_asset_upload_file':
            root = os.environ.get('SPEECH_UPLOAD_ROOT')
            if not root: raise Failure('forbidden','Client-host filesystem upload is not configured.')
            if set(arguments) != {'path'} or not isinstance(arguments['path'], str) or not 0 < len(arguments['path']) <= 4096:
                raise Failure('invalid_request','Supply one relative client-local path.')
            body = await asyncio.to_thread(transport.upload_file, arguments['path'], root=root)
        elif name in ('speech_status', 'speech_skeleton_status'):  # deprecated dispatch alias, not advertised
            if arguments: raise Failure('invalid_request', 'No status arguments accepted.', 'request')
            body = await asyncio.to_thread(transport.get, '/v1/status')
        elif name == 'speech_operation_detail':
            if set(arguments) != {'operation_id'} or not isinstance(arguments['operation_id'], str) or len(arguments['operation_id']) > 128:
                raise Failure('invalid_request', 'Supply one operation_id.', 'operation_id')
            body = await asyncio.to_thread(transport.get, '/v1/operations/' + quote(arguments['operation_id'], safe=''))
        else:
            operation = next((op for op in dict.fromkeys((*IMPLEMENTED, *VOICE_READS, *PRODUCTION_OPERATIONS)) if OPERATIONS[op]['mcp'] == name), None)
            if operation is None: raise Failure('not_found', 'Tool unavailable.', 'tool')
            arguments = dict(arguments)
            key = arguments.pop('idempotency_key', None) if OPERATIONS[operation]['idempotency'] else None
            validate(OPERATIONS[operation]['request'], arguments)
            if operation == 'asset_content' and arguments['end_exclusive'] - arguments['start'] > DIRECT_BYTES:
                raise Failure('resource_limit', 'Direct text MCP range exceeds 3072 decoded bytes.', 'range',
                              'Use python -m clients.cli asset content ID --output FILE outside model context.')
            body = await asyncio.to_thread(transport.call, operation, arguments, **({'key':key} if OPERATIONS[operation]['idempotency'] else {}))
        return result(body, name in ('speech_enroll','speech_speak') and body.get('state') != 'succeeded')
    except Failure as exc:
        return result(exc.body, True)
    except (ConfigurationFailure, InstallationFailure) as exc:
        return result(exc.body, True)
    except (ValidationError, ContractError, TypeError, ValueError):
        return result(problem('invalid_request', 'Request does not match named canonical schema.', 'request'), True)
    except ConnectionError:
        return result(problem('not_ready', 'HTTP transport unavailable.', action='Start the configured HTTP service.'), True)
    except OSError as exc:
        if network_os_error(exc):
            return result(problem('not_ready', 'HTTP transport unavailable.', action='Start the configured HTTP service.'), True)
        return result(problem('invalid_request', 'Local installation or media unavailable.', 'request'), True)

async def main():
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())

if __name__ == '__main__':
    asyncio.run(main())
