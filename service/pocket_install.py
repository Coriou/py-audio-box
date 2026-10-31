"""Closed installed Pocket runtime identity; no imports of installed ML packages.

These are existing requirements/pocket.lock pins, not measured compatibility.
The alternative worker's hash-locked closure retains these same versions.
"""

def runtime_versions():
    return {'pocket-tts': '3.3.0', 'torch': '2.10.0+cpu', 'tokenizers': '0.23.3'}


def manifest(value):
    """Runtime manifest produced by provisioning.bundle_manifest, without policy config."""
    import re
    # Provisioning has already localized config and pinned every file.
    fields = {'schema_version','backend','language','checkpoint','revision','license','access','versions','files'}
    if type(value) is not dict or set(value) != fields or value['backend'] != 'pocket-tts':
        raise ValueError('installed_pocket_manifest')
    if type(value['schema_version']) is not int or value['schema_version'] != 1 or value['language'] not in ('fr','en'):
        raise ValueError('installed_pocket_manifest')
    if value['versions'] != runtime_versions(): raise ValueError('installed_pocket_runtime_versions')
    if not isinstance(value['revision'],str) or not re.fullmatch('[0-9a-f]{40}',value['revision']):
        raise ValueError('installed_pocket_revision')
    if any(not isinstance(value[k],str) or not value[k].strip() for k in ('checkpoint','license','access')):
        raise ValueError('installed_pocket_manifest')
    if type(value['files']) is not list or not 2 <= len(value['files']) <= 128:
        raise ValueError('installed_pocket_manifest')
    names = set(); total = 0
    for entry in value['files']:
        if type(entry) is not dict or set(entry) != {'path','sha256','bytes','origin','revision'}:
            raise ValueError('installed_pocket_file')
        name = entry['path']
        if (not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}', name) or
                not name.endswith(('.yaml','.json','.safetensors')) or name in names or
                type(entry['bytes']) is not int or not 0 < entry['bytes'] <= 2*1024**3 or
                not isinstance(entry['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',entry['sha256']) or
                not isinstance(entry['origin'],str) or not entry['origin'].startswith('https://') or
                entry['revision'] != value['revision']):
            raise ValueError('installed_pocket_file')
        from urllib.parse import urlsplit
        origin=urlsplit(entry['origin'])
        if origin.scheme!='https' or not origin.hostname or origin.username or origin.password or origin.query or origin.fragment:
            raise ValueError('installed_pocket_origin')
        total += entry['bytes']
        if total > 4*1024**3: raise ValueError('installed_pocket_bytes')
        names.add(name)
    if 'config.yaml' not in names or not any(n.endswith('.safetensors') for n in names):
        raise ValueError('installed_pocket_config')
    return value
