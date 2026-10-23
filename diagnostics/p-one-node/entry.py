"""Future ONE-node entry; source review and explicit runtime authority required."""
import os
import sys
from pathlib import Path
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
