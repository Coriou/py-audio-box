"""Pytest integration for the test-only maintenance wait guard."""
import pytest
from _maintenance_wait_guard import MaintenanceWaitGuard, install, activate

def pytest_addoption(parser):
    parser.addoption(
        '--run-manual-b-probes', action='store_true', default=False,
        help='run manual B observer execution probes (requires frozen diagnostic authority)',
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-manual-b-probes'):
        return
    manual = pytest.mark.skip(
        reason='MANUAL/unrun B diagnostic probe; opt in with --run-manual-b-probes',
    )
    for item in items:
        if item.get_closest_marker('manual_b_probe') is not None:
            item.add_marker(manual)

@pytest.fixture(autouse=True)
def maintenance_wait_guard(request, monkeypatch, tmp_path_factory):
    if request.node.path.parent.name not in ('service', 'release'):
        yield None
        return
    guard = MaintenanceWaitGuard(request.node, request.session)
    guard.report_hook = pytest_runtest_makereport
    request.node._maintenance_guard = guard
    install(request, monkeypatch, guard)
    with activate(guard, tmp_path_factory.mktemp('maintenance-guard')):
        yield guard
    if guard.owned:
        guard.fail('missing terminal maintenance receipt at teardown')

@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    guard = getattr(item, '_maintenance_guard', None)
    if guard and guard.failures:
        report.outcome = 'failed'
        report.longrepr = '\n'.join(guard.failures)
    return report
