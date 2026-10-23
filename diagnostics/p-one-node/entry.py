"""Future ONE-node entry; source review and explicit runtime authority required."""
import os
import sys
from pathlib import Path
import hashlib
import importlib.util
import json
import stat


def bootstrap(filename):
    entry = Path(filename).resolve(strict=True)
    root, diagnostics = entry.parents[2], entry.parent
    manifest = json.loads((root / 'manifest.json').read_bytes())
    for name in ('service/__init__.py', 'service/recovery.py'):
        path = root / name
        assert stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), 'STOP regular service source'
        assert path.resolve().parent == root / 'service', 'STOP service root'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['files'][name], 'STOP service digest'
    assert not any(n == 'service' or n.startswith('service.') for n in sys.modules), 'STOP preloaded service'
    sys.path[:] = [str(root), str(diagnostics)] + [p for p in sys.path if p not in (str(root), str(diagnostics))]
    spec = importlib.util.find_spec('service')
    assert spec is not None and spec.origin == str(root / 'service/__init__.py'), 'STOP service origin'
    assert list(spec.submodule_search_locations) == [str(root / 'service')], 'STOP service locations'
    return root


bootstrap(__file__)
import pytest
from one_node_observer import NODE

class Selection:
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session, config, items):
        assert [item.nodeid for item in items] == [NODE], 'STOP selection before fixtures'
        assert items[0].obj.__name__ == NODE.split('::')[1]

if sys.argv[1:] != ['--one-node']:
    raise SystemExit('STOP explicit one-node argument required')
raise SystemExit(pytest.main(['-q', '-x', '--tb=short', NODE], plugins=[Selection()]))
