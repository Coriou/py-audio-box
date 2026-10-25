"""Read-only canonical 1.3.0 catalogue; no backend imports or model inspection."""
import copy
import json
from pathlib import Path
from contracts.validate import SCHEMA, validate

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'contracts/fixtures/v1'
CATALOGUE = json.loads((ROOT / 'contracts/v1/operations.json').read_text())
OPERATIONS = {op['id']: op for op in CATALOGUE['operations']}
IMPLEMENTED = ('discover', 'voices_list', 'asset_get', 'asset_content')
ERROR_STATUS = json.loads((ROOT / 'contracts/v1/errors.json').read_text())['http_status']
VOICE_READS = ('voices_list', 'voice_get', 'voice_version_get', 'voice_versions_list')
ASSET_OPERATIONS = ('asset_upload', 'asset_get', 'asset_content')
PRODUCTION_OPERATIONS = ('audio_export', 'voice_preset_register', 'voice_prepare', 'voice_results', 'voice_save', 'speech_generate', 'job_get', 'job_cancel', 'artifact_get', 'jobs_list', 'job_events', 'job_wait', 'job_retry', 'voice_aliases', 'voice_archive', 'profile_get')
# Artifact reads retain both converter identity and export provenance. Each
# converter registration is independently capped at 16 KiB; preserve both
# copies plus the ordinary 12 KiB envelope within this finite read-only budget.
PRODUCTION_RESPONSE_BYTES = 12 * 1024
ARTIFACT_RESPONSE_BYTES = 64 * 1024
VOICE_RESPONSE_BYTES = 2048
DIRECT_BYTES = 3072
HTTP_BYTES = 16 * 1024 * 1024


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def schema(kind):
    # Include only reachable definitions, never the entire 37-operation schema.
    result = copy.deepcopy(SCHEMA['$defs'][kind])
    definitions = {}
    def visit(value):
        if isinstance(value, dict):
            if '$ref' in value:
                name = value['$ref'].removeprefix('#/$defs/')
                if name not in definitions:
                    definitions[name] = copy.deepcopy(SCHEMA['$defs'][name])
                    visit(definitions[name])
            for key, child in value.items():
                if key != '$ref': visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(result)
    if definitions: result['$defs'] = definitions
    return result


def problem(code, message, field='', action='Read /skeleton; only synthetic fixture reads are available.'):
    return validate('error', dict(code=code, message=message, field=field, action=action,
                                 current_revision=0, changed_ids=[]))


class Failure(Exception):
    def __init__(self, code, message, field='', action='Read /skeleton; only synthetic fixture reads are available.'):
        self.body = problem(code, message, field, action)
        self.status = ERROR_STATUS[code]


def operation_detail(operation_id):
    if operation_id not in OPERATIONS:
        raise Failure('not_found', 'Unknown catalogue operation.', 'operation_id')
    op = OPERATIONS[operation_id]
    return {'fixture': True, 'availability': 'implemented_fixture' if operation_id in IMPLEMENTED else 'unavailable',
            'operation': op, 'request_schema': schema(op['request']), 'response_schema': schema(op['response'])}


def capabilities():
    # source_types is a canonical const naming intended sources, not installed importers.
    # No profile is qualified/installed; detailed unqualified examples are opt-in.
    return validate('capabilities', dict(schema_version='1.3.0', profiles=[],
        operations=list(IMPLEMENTED), engines=[], source_types=['upload','microphone','https','youtube'],
        checked_at='2026-10-16T00:00:00Z'))


def fixture_assets():
    project = json.loads((FIXTURES / 'production-30.json').read_text())
    return {a['id']: a for a in project['assets']}


ASSETS = fixture_assets()
# An explicit manifest, never a caller-controlled filesystem path.
ASSET_FILES = {name: FIXTURES / (name + '.wav') for name in ('impulse', 'sine', 'short', 'long')}
