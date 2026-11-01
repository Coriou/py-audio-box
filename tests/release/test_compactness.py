from tests.release.measure import measure
from service.catalogue import FIXTURES, compact
import json
from contracts.validate import validate
from clients.mcp import tools


def test_measured_catalogue_and_fixture_guardrails():
    advertised = [tool.model_dump(mode='json', exclude_none=True) for tool in tools()]
    assert all(tool.get('description') for tool in advertised if 'description' in tool)
    assert {'speech_enroll', 'speech_speak'} <= {tool['name'] for tool in advertised}
    data = measure()
    # Full headless control now advertises 25 named, typed tools (jobs, voices,
    # profiles and export included), rather than the earlier read-only skeleton.
    assert data['initial_connection_tokens'] <= 3500
    metrics = data['metrics']
    assert metrics['skeleton_status']['utf8_bytes'] <= 2048  # duplicated SDK envelope
    # Canonical raw payload ceilings; envelope duplication is recorded separately.
    exchanges = {e['operation']:e['response'] for e in json.loads((FIXTURES/'transcript-speech_generate.json').read_text())['exchanges']}
    job = exchanges['job_get']
    validate('job',job)
    assert len(compact(job).encode()) <= 1024
    receipt = exchanges['speech_generate']
    validate('receipt',receipt)
    assert len(compact(receipt).encode()) <= 2048
    validate('list', {'items':['fixture-'+str(i) for i in range(20)],'next_cursor':None})
