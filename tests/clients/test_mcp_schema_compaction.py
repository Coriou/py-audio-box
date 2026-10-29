"""Advertised schema compaction must preserve canonical Draft 2020-12 validation."""
import copy
import json

from jsonschema import Draft202012Validator

from clients.mcp import inline_single_use_definitions, input_schema
from service.catalogue import FIXTURES, OPERATIONS, schema


def canonical_input_schema(operation):
    value = schema(OPERATIONS[operation]['request'])
    if OPERATIONS[operation]['idempotency']:
        value['properties']['idempotency_key'] = {
            'type': 'string', 'pattern': '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'}
        value['required'].append('idempotency_key')
    return value


def test_advertised_request_schemas_preserve_canonical_validation():
    transcript = json.loads((FIXTURES / 'transcript-speech_generate.json').read_text())
    identity = transcript['context']['plan']['request']['identity']
    requests = {
        'voice_prepare': {
            'asset_id': 'asset0', 'in_tick': 0, 'out_tick': 100,
            'reference_text': 'reference', 'identity': identity,
            'idempotency_key': 'key0'},
        'speech_generate': {
            **transcript['context']['plan']['request'], 'idempotency_key': 'key0'},
    }
    for operation, valid in requests.items():
        canonical = canonical_input_schema(operation)
        advertised = input_schema(operation)
        Draft202012Validator.check_schema(canonical)
        Draft202012Validator.check_schema(advertised)
        assert '$defs' not in advertised
        assert len(json.dumps(advertised, separators=(',', ':'))) < len(json.dumps(canonical, separators=(',', ':')))
        assert Draft202012Validator(canonical).is_valid(valid)
        cases = [valid]
        for field, bad in (
            ('identity', {**identity, 'engine': 'unknown'}),
            ('identity', {**identity, 'device': 'unknown'}),
            ('identity', {**identity, 'bundle_sha256': 'short'}),
            ('identity', {**identity, 'extra': True}),
            ('device_override', {'family': 'unknown', 'index': 0}),
            ('device_override', {'family': 'cpu', 'index': 256}),
            ('device_override', {'family': 'cpu'}),
        ):
            case = copy.deepcopy(valid)
            case[field] = bad
            cases.append(case)
        for field in ('identity', 'idempotency_key'):
            case = copy.deepcopy(valid)
            del case[field]
            cases.append(case)
        if operation == 'speech_generate':
            cases.append({**valid, 'text_rendering': {
                'profile': 'numeric-v1', 'version': 1, 'locale': 'unknown',
                'annotations': []}})
        assert all(not Draft202012Validator(canonical).is_valid(case) for case in cases[1:])
        for case in cases:
            assert Draft202012Validator(advertised).is_valid(case) == Draft202012Validator(canonical).is_valid(case)


def test_unknown_cyclic_and_sibling_refs_remain_untouched():
    value = {'type': 'object', 'properties': {
        'unknown': {'$ref': '#/$defs/missing'},
        'sibling': {'$ref': '#/$defs/single', 'description': 'keep sibling'},
        'cyclic': {'$ref': '#/$defs/cycle'}},
        '$defs': {'single': {'type': 'string'}, 'cycle': {'$ref': '#/$defs/cycle'}}}
    assert inline_single_use_definitions(value) == value


def test_nested_single_use_definition_keeps_retained_definition_valid():
    canonical = {'$ref': '#/$defs/outer', '$defs': {
        'outer': {'type': 'object', 'properties': {'field': {'$ref': '#/$defs/inner'}},
                  'required': ['field'], 'additionalProperties': False},
        'inner': {'type': 'string', 'minLength': 2}}}
    advertised = inline_single_use_definitions(canonical)
    assert advertised == {'$ref': '#/$defs/outer', '$defs': {
        'outer': {'type': 'object', 'properties': {'field': {'type': 'string', 'minLength': 2}},
                  'required': ['field'], 'additionalProperties': False}}}
    for case in ({'field': 'ok'}, {'field': 'x'}, {'field': 2}, {'field': 'ok', 'extra': True}, {}):
        assert Draft202012Validator(advertised).is_valid(case) == Draft202012Validator(canonical).is_valid(case)


def test_uri_and_anchor_scope_stays_unchanged():
    for keyword, marker in (('$id', 'https://example.test/schema'),
                            ('$anchor', 'named'), ('$dynamicRef', '#named')):
        value = {'$ref': '#/$defs/inner', '$defs': {
            'inner': {'type': 'string', keyword: marker}}}
        assert inline_single_use_definitions(value) == value
