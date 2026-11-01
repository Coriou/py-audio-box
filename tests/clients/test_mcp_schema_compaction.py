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


def test_empty_discover_schema_compaction_preserves_validation(monkeypatch):
    from clients import mcp
    canonical = canonical_input_schema('discover')
    advertised = input_schema('discover')
    assert advertised == {'type': 'object', 'maxProperties': 0}
    for value in ({}, {'unexpected': 1}, [], None, 0, ''):
        assert Draft202012Validator(advertised).is_valid(value) == Draft202012Validator(canonical).is_valid(value)
    # Future canonical constraints must survive instead of being dropped by compaction.
    constrained = {**canonical, 'description': 'keep', 'minProperties': 0}
    monkeypatch.setattr(mcp, 'schema', lambda kind: copy.deepcopy(constrained))
    assert input_schema('discover') == constrained


def test_repeated_identifier_refs_preserve_validation():
    for operation in ('voice_prepare', 'voice_save', 'speech_generate'):
        canonical = canonical_input_schema(operation)
        advertised = input_schema(operation)
        assert advertised.get('$defs', {}).get('i') == {
            'type': 'string', 'pattern': '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'}
        Draft202012Validator.check_schema(advertised)
        for value in ({}, {'idempotency_key': 'ok'}, {'idempotency_key': 'bad key'}, []):
            assert Draft202012Validator(advertised).is_valid(value) == Draft202012Validator(canonical).is_valid(value)


def test_input_schema_never_rewrites_literal_data(monkeypatch):
    from clients import mcp
    identifier = {'type': 'string', 'pattern': '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'}
    literal_ref = {'$ref': '#/$defs/payload'}
    canonical = {'type': 'object', 'properties': {
        'a': copy.deepcopy(identifier), 'b': copy.deepcopy(identifier),
        'marker': {'const': copy.deepcopy(identifier)},
        'choice': {'enum': [copy.deepcopy(identifier)]},
        'fallback': {'type': 'object', 'default': copy.deepcopy(identifier)},
        'literal_ref': {'const': copy.deepcopy(literal_ref)},
        'payload': {'$ref': '#/$defs/payload'}},
        'required': ['a', 'b', 'marker', 'choice', 'literal_ref', 'payload'],
        'additionalProperties': False,
        '$defs': {'payload': {'type': 'string'}}}
    monkeypatch.setattr(mcp, 'schema', lambda kind: copy.deepcopy(canonical))
    advertised = input_schema('voice_prepare')
    assert advertised['properties']['marker']['const'] == identifier
    assert advertised['properties']['choice']['enum'] == [identifier]
    assert advertised['properties']['fallback']['default'] == identifier
    assert advertised['properties']['literal_ref']['const'] == literal_ref
    assert advertised['properties']['payload'] == {'type': 'string'}
    assert advertised['properties']['a'] == {'$ref': '#/$defs/i'}
    expected = copy.deepcopy(canonical)
    expected['properties']['idempotency_key'] = copy.deepcopy(identifier)
    expected['required'].append('idempotency_key')
    valid = {'a': 'one', 'b': 'two', 'idempotency_key': 'key',
             'marker': identifier, 'choice': identifier,
             'literal_ref': literal_ref, 'payload': 'ok'}
    for value in (valid, {**valid, 'marker': 'wrong'},
                  {**valid, 'choice': 'wrong'}, {**valid, 'a': 1}):
        assert Draft202012Validator(advertised).is_valid(value) == Draft202012Validator(expected).is_valid(value)


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
        assert advertised.get('$defs', {}).keys() == {'i'}
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
