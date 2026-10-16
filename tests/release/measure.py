"""Named deterministic serialized client fixture; not actual host/model acceptance."""
import json
import os
from pathlib import Path
import tiktoken
from clients.mcp import tools, result
from service.catalogue import FIXTURES, capabilities, compact
from service.api import summary


def measure():
    encoding = tiktoken.get_encoding('o200k_base')
    # Explicit offline receipt/status fixtures; these are NOT available job endpoints.
    transcript = json.loads((FIXTURES/'transcript-speech_generate.json').read_text())
    exchanges = {e['operation']:e['response'] for e in transcript['exchanges']}
    receipt = exchanges['speech_generate']
    status = exchanges['job_get']
    items = {'items':['fixture-'+str(i) for i in range(20)],'next_cursor':None}
    values = {'tools_list': {'tools':[t.model_dump(mode='json',exclude_none=True) for t in tools()]},
              'discovery_result':result(capabilities()).model_dump(mode='json',exclude_none=True),
              'skeleton_status':result(summary()).model_dump(mode='json',exclude_none=True),
              'receipt_fixture':result(receipt).model_dump(mode='json',exclude_none=True),
              'job_status_fixture':result(status).model_dump(mode='json',exclude_none=True),
              'list_20_fixture':result(items).model_dump(mode='json',exclude_none=True)}
    metrics = {}
    for name, value in values.items():
        serialized = compact(value)
        metrics[name] = {'utf8_bytes':len(serialized.encode()),'tokens':len(encoding.encode(serialized))}
    raw = {'receipt':receipt,'job_status':status,'list_20':items}
    for name, value in raw.items():
        text = compact(value)
        metrics[name+'_raw'] = {'utf8_bytes':len(text.encode()),'tokens':len(encoding.encode(text))}
    return {'fixture':'0C SDK tools/list + CallToolResult compact UTF-8, text + structuredContent counted',
        'tokenizer':'tiktoken 0.14.0 / o200k_base','agent_model':'none (offline client fixture)',
        'host_duplication':'text and structuredContent included once each; other host prompts/copies unknown',
        'metrics':metrics,
        'initial_connection_tokens':metrics['tools_list']['tokens']+metrics['discovery_result']['tokens']}

if __name__ == '__main__': print(json.dumps(measure(),indent=2))
