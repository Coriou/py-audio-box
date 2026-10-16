"""MCP v1 maintenance SDK, stdio; named reads call HTTP, never a command bus."""
import asyncio
from urllib.parse import quote
from jsonschema.exceptions import ValidationError
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult
from contracts.validate import ContractError, validate
from clients.http import HTTP
from service.catalogue import (DIRECT_BYTES, Failure, IMPLEMENTED, OPERATIONS, compact, problem, schema)

server = Server('speech-synthetic-0c')


def tools():
    items = [Tool(name=OPERATIONS[op]['mcp'],
        description=('Synthetic read-only fixture; no service store or ML. ' +
            ('Explicit opt-in bytes only; at most 3072 decoded bytes. Larger downloads: python -m clients.cli asset content ID --output FILE.' if op == 'asset_content' else op)),
        inputSchema=schema(OPERATIONS[op]['request'])) for op in IMPLEMENTED]
    items += [Tool(name='speech_skeleton_status', description='Compact synthetic skeleton status and limitations.',
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
        transport = HTTP()
        if name == 'speech_skeleton_status':
            if arguments: raise Failure('invalid_request', 'No status arguments accepted.', 'request')
            body = await asyncio.to_thread(transport.get, '/skeleton')
        elif name == 'speech_operation_detail':
            if set(arguments) != {'operation_id'} or not isinstance(arguments['operation_id'], str) or len(arguments['operation_id']) > 128:
                raise Failure('invalid_request', 'Supply one operation_id.', 'operation_id')
            body = await asyncio.to_thread(transport.get, '/skeleton/operations/' + quote(arguments['operation_id'], safe=''))
        else:
            operation = next((op for op in IMPLEMENTED if OPERATIONS[op]['mcp'] == name), None)
            if operation is None: raise Failure('not_found', 'Tool unavailable in the 0C skeleton.', 'tool')
            validate(OPERATIONS[operation]['request'], arguments)
            if operation == 'asset_content' and arguments['end_exclusive'] - arguments['start'] > DIRECT_BYTES:
                raise Failure('resource_limit', 'Direct text MCP range exceeds 3072 decoded bytes.', 'range',
                              'Use python -m clients.cli asset content ID --output FILE outside model context.')
            body = await asyncio.to_thread(transport.call, operation, arguments)
        return result(body)
    except Failure as exc:
        return result(exc.body, True)
    except (ValidationError, ContractError, TypeError, ValueError):
        return result(problem('invalid_request', 'Request does not match named canonical schema.', 'request'), True)
    except ConnectionError:
        return result(problem('not_ready', 'Skeleton HTTP transport unavailable.', action='Start the configured loopback fixture HTTP service.'), True)

async def main():
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())

if __name__ == '__main__':
    asyncio.run(main())
