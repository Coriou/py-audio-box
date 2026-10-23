"""Strict version-1 runtime receipt; only terminal ldd load addresses are observational."""
import json
import re


def require(ok, reason):
    if not ok:
        raise RuntimeError('STOP identity ' + reason)


def parse_ldd(raw):
    text = raw.decode('ascii', errors='strict')
    require(text.endswith('\n'), 'ldd newline')
    records, names = [], set()
    for line in text.splitlines():
        # Three supported Linux ldd forms, with exactly one terminal ASLR address.
        match = re.fullmatch(r'\s*(linux-vdso\.so\.\d+) \(0x[0-9a-f]+\)', line)
        if match:
            record = ['vdso', match[1], None]
        else:
            match = re.fullmatch(r'\s*([A-Za-z0-9_.+-]+) => (/[^\s()]+) \(0x[0-9a-f]+\)', line)
            if match:
                record = ['library', match[1], match[2]]
            else:
                match = re.fullmatch(r'\s*(/[^\s()]+/ld-linux[^/\s()]+) \(0x[0-9a-f]+\)', line)
                require(match is not None, 'unsupported ldd line')
                record = ['loader', match[1], match[1]]
        require(record[1] not in names, 'duplicate ldd name')
        names.add(record[1])
        records.append(record)
    require(sum(r[0] == 'vdso' for r in records) == 1 and
            sum(r[0] == 'loader' for r in records) == 1 and
            any(r[0] == 'library' and r[1] == 'libsqlite3.so.0' for r in records), 'incomplete ldd')
    return records  # Preserve order and every semantic byte; duplicates are forbidden.


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


FIELDS = {'python', 'implementation', 'version', 'optimization', 'assertions', 'pins',
          'sqlite', 'source_id', 'compile_options', 'extension', 'ldd', 'library', 'dpkg', 'commands'}


def semantic(raw):
    data = json.loads(raw, object_pairs_hook=pairs)
    require(set(data) == {'schema', 'semantic', 'raw_ldd'} and type(data['schema']) is int and data['schema'] == 1, 'schema')
    value = data['semantic']
    require(isinstance(value, dict) and set(value) == FIELDS, 'fields')
    require(isinstance(data['raw_ldd'], str) and
            parse_ldd(data['raw_ldd'].encode('ascii')) == value['ldd'], 'raw ldd binding')
    require(isinstance(value['commands'], dict) and set(value['commands']) == {'ldd', 'dpkg'} and
            all(isinstance(v, dict) and set(v) == {'returncode', 'stderr'} and
                type(v['returncode']) is int and isinstance(v['stderr'], str) for v in value['commands'].values()), 'command shape')
    require(value['commands'] == {'ldd': {'returncode': 0, 'stderr': ''},
                                  'dpkg': {'returncode': 0, 'stderr': ''}}, 'command status')
    require(value['assertions'] is True and type(value['optimization']) is int and
            value['optimization'] == 0, 'assertions')
    require(all(isinstance(value[k], str) and value[k] for k in
                ('python', 'implementation', 'sqlite', 'source_id', 'extension', 'dpkg')), 'string fields')
    require(value['extension'].startswith('/') and value['implementation'] == 'cpython', 'runtime paths')
    require(isinstance(value['version'], list) and len(value['version']) == 5 and
            all(type(v) is int for v in value['version'][:3]) and
            isinstance(value['version'][3], str) and type(value['version'][4]) is int, 'version')
    pins = value['pins']
    require(isinstance(pins, list) and len(pins) == 42 and all(
        isinstance(p, list) and len(p) == 3 and all(isinstance(x, str) and x for x in p)
        and p[1] == p[2] for p in pins), 'pins')
    require(len({p[0].lower().replace('_', '-') for p in pins}) == 42, 'duplicate pins')
    options = value['compile_options']
    require(isinstance(options, list) and bool(options) and all(
        isinstance(p, list) and len(p) == 1 and isinstance(p[0], str) and p[0] for p in options)
        and len({p[0] for p in options}) == len(options), 'compile options')
    library = value['library']
    require(isinstance(library, dict) and set(library) == {'path', 'sha256'} and
            isinstance(library['path'], str) and library['path'].startswith('/') and
            isinstance(library['sha256'], str) and re.fullmatch('[0-9a-f]{64}', library['sha256']), 'library')
    return value


def compare(before, after):
    require(semantic(before) == semantic(after), 'semantic drift')
