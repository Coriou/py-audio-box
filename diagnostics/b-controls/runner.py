"""Finite B controls only. HOST may import this module: stdlib, no dispatch on import."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
CONTROL = 'tests/service/test_child_wait_guard.py'
GUARD = 'd3a011ffcc50e785f39abba54f07b51fe733b132425fff8dfc9a9848c92d254d'
CONTROLS = 'fbea91ee219305f934b319581f58b0b3ad3b6839abbbcf877298751a1c9512db'
GROUPS = (
    ('test_B_closed_transcript_source_predicate', 'B-T-', 'B_CONTROL_PARSER', 17, 10),
    ('test_B_directed_isolated_seams', 'B-S-', 'B_CONTROL_SEAMS', 39, 35),
    ('test_B_parent_source_predicates', 'B-P-', 'B_CONTROL_PARENT', 23, 10),
    ('test_B_complete_receipt_ownership', 'B-O-', None, 2, 10),
    ('test_B_fresh_real_inner_binding_control_sink', 'B-H-', None, 2, 45),
    ('test_B_timeout_shape_source_predicate', 'B-X-', None, 4, 10),
    ('test_B_precapture_same_collected_function', 'B-C-', 'B_CONTROL_B1', 6, 65),
    ('test_B_independent_authority_source_predicates', 'B-A-', 'B_CONTROL_AUTHORITY', 7, 10),
)
TOTAL = 2600                 # 2375 node caps + pre60 + reap30 + post60 + margin75
DIRECTED_INDEX = 18
DIRECTED_NODE = CONTROL + '::test_B_directed_isolated_seams[B-S-collected-control]'
DIRECTED_TOTAL = 260         # pre60 + node35 + reap30 + post60 + margin75
COMPLEMENT_INDICES = tuple(range(17)) + tuple(range(18, 100))
COMPLEMENT_ID = 'existing100-minus017-v1'
# Caps are authenticated by inventory before this closed selection is consumed.
COMPLEMENT_NODE_BUDGET = sum(count * cap for _, _, _, count, cap in GROUPS) - 35
COMPLEMENT_TOTAL = 60 + COMPLEMENT_NODE_BUDGET + 30 + 60 + 75
STREAM = 16 * 1024**2       # existing finite-driver capture cap, each stream


def require(ok, why):
    if not ok:
        raise RuntimeError('STOP B ' + why)


def helpers():
    spec = importlib.util.spec_from_file_location('b_finite_driver', ROOT/'diagnostics/p-one-node/finite-driver.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SEPARATE_HELPER = 'test_B_overhead_extracted_HOST_contract'
SEPARATE_HELPER_AST = '2cf539696bc0f0fb9b03b1869a775a2dfe138a12e77a4f3e1d615dd5eb5cf6d9'


def inventory(root=ROOT):
    """Read literal params and ID lambda; never import controls or collect pytest."""
    raw = (root/CONTROL).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CONTROLS, 'controls digest')
    tree = ast.parse(raw)
    return inventory_tree(tree)


def inventory_tree(tree):
    """Closed AST inventory; inventory() first authenticates complete source bytes."""
    assignments = {n.targets[0].id: n.value for n in tree.body
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_B_')]
    require([n.name for n in definitions] == [g[0] for g in GROUPS] + [SEPARATE_HELPER],
            'eight definitions/order plus exact separate helper')
    require(hashlib.sha256(ast.dump(definitions[-1], include_attributes=False).encode()).hexdigest()
            == SEPARATE_HELPER_AST, 'separate helper AST')
    definitions = definitions[:-1]
    rows = []
    for definition, (name, prefix, variable, count, cap) in zip(definitions, GROUPS):
        require(len(definition.decorator_list) == 1, 'decorator count')
        decorator = definition.decorator_list[0]
        require(isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == 'pytest.mark.parametrize'
                and len(decorator.args) == 2 and ast.literal_eval(decorator.args[0]) == 'control', 'parametrize shape')
        params = decorator.args[1]
        if variable:
            require(isinstance(params, ast.Name) and params.id == variable, 'parameter authority')
            params = assignments[variable]
        values = ast.literal_eval(params)
        require(type(values) is tuple and len(values) == count and all(type(v) is str for v in values), 'finite literals')
        require(len(decorator.keywords) == 1 and decorator.keywords[0].arg == 'ids', 'ID keyword')
        ident = decorator.keywords[0].value
        require(isinstance(ident, ast.Lambda) and ast.unparse(ident) == 'lambda c: '+repr(prefix)+' + c', 'ID lambda')
        for value in values:
            rows.append(dict(id=prefix+value, node=CONTROL+'::'+name+'['+prefix+value+']', seconds=cap))
    require(len(rows) == len({r['node'] for r in rows}) == 100, 'exact unique100')
    require(sum(r['seconds'] for r in rows) == 2375, 'node budget sum')
    return rows


def directed_selection(root=ROOT):
    # Authenticate the entire closed100 before selecting the sole approved row.
    rows = inventory(root)
    row = rows[DIRECTED_INDEX]
    require(row == dict(id='B-S-collected-control', node=DIRECTED_NODE, seconds=35),
            'closed directed018 selection')
    return [(DIRECTED_INDEX, row)]


def complement_selection(root=ROOT):
    rows = inventory(root)  # full source digest AND closed100 AST first
    require(rows[17] == dict(id='B-S-original74-real-observer', node=CONTROL+'::test_B_directed_isolated_seams[B-S-original74-real-observer]', seconds=35),
            'excluded017 identity')
    selected = [(i, rows[i]) for i in COMPLEMENT_INDICES]
    require(len(selected) == 99 and sum(row['seconds'] for _, row in selected)
            == COMPLEMENT_NODE_BUDGET, 'closed complement caps')
    return selected


def environment(tmp):
    # Inherit locked image environment (including tokenizer cache), remove test
    # selection/optimization controls; identical root works for nested -c children.
    env = dict(os.environ)
    for name in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PYTHONOPTIMIZE', 'PYTHONSTARTUP'):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        PYTHONPATH=os.pathsep.join(map(str, (ROOT, ROOT/'tests', ROOT/'diagnostics/p-one-node'))),
        TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp), PAB_MAINTENANCE_WAIT_LOG='/tmp/waits.jsonl')
    return env


def node_argv(index, directory):
    return [sys.executable, '-B', str(ROOT/'diagnostics/b-controls/runner.py'),
            'node', str(index), str(directory)]


def pytest_argv(row, directory):
    return ['--noconftest', '-p', 'no:cacheprovider', '-q', '-x', '--tb=short',
            '--rootdir='+str(ROOT), '--basetemp='+str(directory/'pytest'), row['node']]


class Selection:
    """Plugin requires exactly one item BEFORE fixtures; every phase must pass."""
    def __init__(self, row, directory, finite):
        self.row, self.directory, self.finite = row, directory, finite
        self.collected = False
        self.reports = []

    def pytest_collection_finish(self, session):
        items = session.items
        require([i.nodeid for i in items] == [self.row['node']], 'selection before fixtures')
        item, = items
        require(Path(item.path).resolve() == ROOT/CONTROL and
                Path(item.obj.__code__.co_filename).resolve() == ROOT/CONTROL, 'collected source path')
        self.collected = True
        with self.finite.private(self.directory/'selection.json') as stream:
            stream.write(json.dumps(dict(node=self.row['node'], count=1, before_fixtures=True)).encode())

    def pytest_runtest_logreport(self, report):
        require(self.collected and report.nodeid == self.row['node'], 'report selection')
        self.reports.append([report.when, report.outcome])
        require(len(self.reports) <= 3, 'report count')

    def accepted(self, rc):
        return rc == 0 and self.collected and self.reports == [
            ['setup', 'passed'], ['call', 'passed'], ['teardown', 'passed']]


def node(index, directory):
    require(sys.platform == 'linux' and ROOT == Path('/app'), 'node Linux /app only')
    f = helpers()
    f.enabled()
    f.verify_source()
    rows = inventory()
    require(0 <= index < len(rows), 'node index')
    require(directory == Path('/tmp/b-controls')/('%03d' % index), 'node directory')
    require(not any(n == 'service' or n.startswith('service.') for n in sys.modules), 'preloaded service')
    sys.path[:0] = [str(ROOT), str(ROOT/'tests'), str(ROOT/'diagnostics/p-one-node')]
    spec = importlib.util.find_spec('service')
    require(spec is not None and spec.origin == str(ROOT/'service/__init__.py')
            and list(spec.submodule_search_locations) == [str(ROOT/'service')], 'service root bootstrap')
    # Import pytest only in this explicitly Linux runtime entry, never HOST.
    import pytest
    plugin = Selection(rows[index], directory, f)
    # trylast collection_finish executes after modifyitems and before runtestloop.
    Selection.pytest_collection_finish = pytest.hookimpl(trylast=True)(Selection.pytest_collection_finish)
    rc = int(pytest.main(pytest_argv(rows[index], directory), plugins=[plugin]))
    accepted = plugin.accepted(rc)
    with f.private(directory/'result.json') as stream:
        stream.write(json.dumps(dict(node=rows[index]['node'], pytest_exit=rc,
            collected=plugin.collected, reports=plugin.reports, accepted=accepted)).encode())
    require(accepted, 'pytest exit/collection/phases')
    f.verify_source()
    return 0


def ordered(root, finite, *, capture=None, clock=time.monotonic, observe=lambda: None, finished=lambda: None, directed=False, complement=False, completed=None):
    """Actual orchestration shared with HOST fake external boundary; no retries."""
    capture = finite.capture if capture is None else capture
    require(not (directed and complement), 'exclusive stage')
    indexed = complement_selection() if complement else directed_selection() if directed else list(enumerate(inventory()))
    budget = sum(row['seconds'] for _, row in indexed)
    start = clock()
    completed = [] if completed is None else completed
    for index, row in indexed:
        observe()
        require(clock() < start + budget, 'campaign deadline before dispatch')
        directory = root/('%03d' % index)
        directory.mkdir(mode=0o700)
        tmp = directory/'tmp'
        tmp.mkdir(mode=0o700)
        record = dict(index=index, **row, state='PENDING')
        finite.slot_create(directory/'receipt.slot')
        finite.slot_write(directory/'receipt.slot', record)
        try:
            deadline = min(start+budget, clock()+row['seconds'])
            rc, pid = capture(node_argv(index, directory),
                [directory/'stdout', directory/'stderr'], deadline,
                env=environment(tmp), observe=observe)
            record.update(exit=rc, pid=pid)
            finished()  # leader reaped; no waitpid competes with an active test
            require(rc == 0, 'first unexpected node exit')
            # Child exit alone cannot pass: verify durable exact selection/result.
            selection = json.loads(finite.read_regular(directory/'selection.json', 4096, 0o600))
            result = json.loads(finite.read_regular(directory/'result.json', 4096, 0o600))
            require(selection == dict(node=row['node'], count=1, before_fixtures=True), 'selection receipt')
            require(result.get('node') == row['node'] and result.get('pytest_exit') == 0
                and result.get('collected') is True and result.get('accepted') is True
                and result.get('reports') == [['setup', 'passed'], ['call', 'passed'], ['teardown', 'passed']], 'result receipt')
            require(clock() < deadline, 'node completed after deadline')
            record['streams'] = [{ 'name': p.name, 'size': p.stat().st_size,
                'sha256': finite.regular_hash(p)} for p in (directory/'stdout', directory/'stderr')]
            record['state'] = 'PASS'
        except BaseException as exc:
            record.update(state='STOP', failure=type(exc).__name__+': '+str(exc)[:256])
            raise
        finally:
            # Partial capture bytes remain even when no completion/receipt possible.
            record['streams'] = [{'name': p.name, 'size': p.stat().st_size,
                'sha256': finite.regular_hash(p)} for p in (directory/'stdout', directory/'stderr') if p.exists()]
            finite.slot_write(directory/'receipt.slot', record)
        completed.append(row['id'])
    return completed


def campaign(token, invocation, *, directed=False, complement=False):
    f = helpers()
    f.enabled()
    require(sys.platform == 'linux' and ROOT == Path('/app') and os.getpid() == 1
            and os.getuid() == os.getgid() == 10001, 'campaign PID1 identity')
    require(re.fullmatch('[0-9a-f]{48}', token) and str(uuid.UUID(invocation)) == invocation, 'owner')
    os.umask(0o077)
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    require(not (directed and complement), 'exclusive stage')
    node_budget = COMPLEMENT_NODE_BUDGET if complement else 35 if directed else 2375
    total = COMPLEMENT_TOTAL if complement else DIRECTED_TOTAL if directed else TOTAL
    signal.setitimer(signal.ITIMER_REAL, total)
    root = Path('/tmp')
    state = dict(state='PENDING', token=token, invocation=invocation,
        expected=99 if complement else 1 if directed else 100, acceptance=False, physical_quota=False)
    if complement:
        state.update(selection_identity=COMPLEMENT_ID, selected_indices=list(COMPLEMENT_INDICES), completed=[])
    if directed:
        state.update(selected_index=DIRECTED_INDEX, selected_node=directed_selection()[0][1]['node'])
    stop = threading.Event()
    errors = []
    terminal = root/'pab-terminal.slot'
    try:
        f.initialized(root)
        memory = {line.split(':')[0]: int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()}
        require(memory['MemAvailable'] >= 4*f.GIB, 'memory admission')
        for name in ('pab-intent.slot', 'pab-terminal.slot'):
            f.slot_create(root/name)
        f.slot_write(root/'pab-intent.slot', state)
        with f.private(root/'retention-sentinel') as stream:
            stream.write(token.encode()+b'\x00\xff\x80\r\n')
        f.verify_source()
        inventory()
        f.checked_program(['python', '-c', f.identity_program()], root/'identity-before', start+60)
        require(time.monotonic() < start+60, 'precheck deadline')
        controls = root/'b-controls'
        controls.mkdir(mode=0o700)
        def monitor():
            while not stop.is_set():
                try: f.census(root)
                except BaseException as exc:
                    errors.append(type(exc).__name__)
                    return
                stop.wait(0.5)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        def observe():
            require(not errors, 'census failed')
            require(time.monotonic() < start+60+node_budget, 'campaign outer deadline')
        state['completed'] = ordered(controls, f, observe=observe,
            finished=lambda: require(not f.namespace_pids(), 'descendants remain after control'), directed=directed, complement=complement, completed=state['completed'] if complement else None)
        require(len(state['completed']) == (99 if complement else 1 if directed else 100), 'completed expected selection')
    except BaseException as exc:
        state['failure'] = type(exc).__name__+': '+str(exc)[:256]
    finally:
        try:
            # No namespace wait races with an active test. After phase ends,
            # existing PID1 cleanup reaps namespace descendants once, no rerun.
            state['descendants'] = f.quiesce(min(start+60+node_budget+30, time.monotonic()+30))
            require(state['descendants']['complete'], 'unresolved descendants')
            stop.set()
            if 'thread' in locals():
                thread.join(timeout=1)
                require(not thread.is_alive(), 'monitor unresolved')
            require(not errors, 'incomplete census')
            f.verify_source()
            end = min(start+60+node_budget+30+60, time.monotonic()+60)
            f.checked_program(['python', '-c', f.identity_program()], root/'identity-after', end)
            f.compare_identity(f.read_regular(root/'identity-before', f.META), f.read_regular(root/'identity-after', f.META))
            require(time.monotonic() < end and 'failure' not in state, 'campaign/check failure')
            state.update(state='TERMINAL', classification='B-CONTROLS-REQUIRES-INDEPENDENT-ACTUAL-REVIEW')
        except BaseException as exc:
            state.update(state='STOP' if state.get('descendants', {}).get('complete') else 'UNKNOWN',
                terminal_failure=type(exc).__name__+': '+str(exc)[:256])
        if complement:
            state['completed_indices'] = list(COMPLEMENT_INDICES[:len(state['completed'])])
        state['elapsed'] = time.monotonic()-start
        try:
            require(time.monotonic() < start+total, 'publication deadline')
            f.slot_write(terminal, state)
        except BaseException:
            state['state'] = 'UNKNOWN'
    return 0 if state['state'] == 'TERMINAL' else 1


def campaign_command(token, invocation, *, directed=False, complement=False):
    require(not (directed and complement), 'exclusive stage')
    require(re.fullmatch('[0-9a-f]{48}', token) and str(uuid.UUID(invocation)) == invocation, 'owner')
    return ['python', '/app/diagnostics/b-controls/runner.py', 'complement099' if complement else 'directed018' if directed else 'campaign', token, invocation]


def create_argv(name, volume, image, token, invocation, *, directed=False, complement=False):
    """Distinct campaign command binding; NEVER Lifecycle.create_argv(writer)."""
    require(re.fullmatch('[0-9a-f]{64}', image) is not None or
            re.fullmatch('sha256:[0-9a-f]{64}', image) is not None, 'full image ID')
    require(re.fullmatch('pab-b-[a-z0-9-]+', name) and re.fullmatch('pab-b-[a-z0-9-]+', volume), 'owned names')
    return ['docker', 'create', '--name', name, '--label', 'pab.owner='+token,
        '--label', 'pab.invocation='+invocation, '--cpus', '2', '--memory', '2g',
        '--pids-limit', '256', '--network', 'none', '--read-only', '--user', '10001:10001',
        '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--restart', 'no',
        '--log-driver', 'none', '--workdir', '/app', '--mount', 'type=volume,src='+volume+',dst=/tmp',
        '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONPATH=/app:/app/tests:/app/diagnostics/p-one-node',
        '-e', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1', '-e', 'PAB_MAINTENANCE_WAIT_LOG=/tmp/waits.jsonl',
        '-e', 'TIKTOKEN_CACHE_DIR=/tokenizer-cache', image]+campaign_command(token, invocation, directed=directed, complement=complement)


def inspect_campaign(obj, name, volume, image, token, invocation, *, directed=False, complement=False):
    """Read-only admission shape for a later separately authorized host owner.

    This does not dispatch anything or adapt the frozen singleton Lifecycle.
    Campaign startup allowlist adds only PYTEST_DISABLE_PLUGIN_AUTOLOAD to
    the frozen launcher policy; its original command binding stays unchanged.
    """
    cfg, hc = obj['Config'], obj['HostConfig']
    require(re.fullmatch('[0-9a-f]{64}', obj['Id']) and obj['Name'] == '/'+name
        and obj['Image'] == image, 'container identity')
    require(cfg['Labels'] == {'pab.owner':token, 'pab.invocation':invocation}
        and cfg['Cmd'] == campaign_command(token, invocation, directed=directed, complement=complement)
        and cfg['User'] == '10001:10001' and cfg['WorkingDir'] == '/app', 'campaign command binding')
    require(not cfg.get('Entrypoint') and not cfg.get('Healthcheck') and not cfg.get('OnBuild') and not cfg.get('Volumes')
        and cfg.get('StopSignal', '') in ('', 'SIGTERM', '15'), 'startup shape')
    expected = {'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=/app:/app/tests:/app/diagnostics/p-one-node',
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1', 'PAB_MAINTENANCE_WAIT_LOG=/tmp/waits.jsonl',
        'TIKTOKEN_CACHE_DIR=/tokenizer-cache'}
    allowed = {'PATH', 'LANG', 'PYTHON_VERSION', 'PYTHON_SHA256', 'PYTHON_PIP_VERSION',
        'PYTHON_SETUPTOOLS_VERSION', 'PYTHON_GET_PIP_URL', 'PYTHON_GET_PIP_SHA256',
        'PYTHONDONTWRITEBYTECODE', 'PYTHONPATH', 'PAB_MAINTENANCE_WAIT_LOG', 'TIKTOKEN_CACHE_DIR',
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD'}
    require(all(e.split('=',1)[0] in allowed for e in cfg['Env'])
        and len({e.split('=',1)[0] for e in cfg['Env']}) == len(cfg['Env']), 'startup environment')
    require(expected <= set(cfg['Env']) and
        len([e for e in cfg['Env'] if e.split('=',1)[0] in {v.split('=',1)[0] for v in expected}]) == len(expected), 'campaign environment')
    require(hc['NanoCpus'] == 2000000000 and hc['Memory'] == 2*1024**3 and hc['PidsLimit'] == 256
        and hc['ReadonlyRootfs'] and hc['NetworkMode'] == 'none' and hc['CapDrop'] == ['ALL']
        and 'no-new-privileges' in hc['SecurityOpt'] and not hc.get('CapAdd')
        and not hc.get('Privileged') and not hc.get('AutoRemove')
        and hc['RestartPolicy']['Name'] == 'no' and hc['LogConfig']['Type'] == 'none', 'security/resources')
    require(not any(hc.get(k) for k in ('Binds','Devices','PortBindings','Tmpfs','VolumesFrom'))
        and not cfg.get('ExposedPorts') and hc.get('PidMode','') == '', 'extra exposure')
    mounts = obj['Mounts']
    require(len(mounts) == 1 and mounts[0]['Type'] == 'volume' and mounts[0]['Name'] == volume
        and mounts[0]['Destination'] == '/tmp' and mounts[0]['RW'] is True, 'fresh campaign volume binding')
    return obj['Id']


def main():
    if sys.argv[1:] == ['inventory']:
        print(json.dumps(inventory(), indent=2))
        return 0
    require(len(sys.argv) == 4, 'fixed CLI')
    if sys.argv[1] in ('campaign', 'directed018', 'complement099'):
        return campaign(*sys.argv[2:], directed=sys.argv[1] == 'directed018', complement=sys.argv[1] == 'complement099')
    require(sys.argv[1] == 'node' and sys.argv[2].isdecimal(), 'node CLI')
    return node(int(sys.argv[2]), Path(sys.argv[3]))


if __name__ == '__main__':
    raise SystemExit(main())
