"""Pytest integration for the test-only maintenance wait guard."""
import pytest
from _maintenance_wait_guard import MaintenanceWaitGuard, install, activate

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
