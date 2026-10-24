"""Bounded stdlib HOST: extracted predicates and synthetic events only.

Never imports/executes guard, product, pytest or its rewriter. Full installation,
fixture teardown and actual trace restoration remain separately pending.
"""
import ast
import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace, FunctionType

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT/'tests/_maintenance_wait_guard.py'
SOURCE = GUARD.read_bytes()
TREE = ast.parse(SOURCE)
B = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'BlockedPipeWitness_B')
NAMES = {'nested', 'code_projection', 'fact_key', 'target_source', 'sites', 'timeout_sites',
         'select_B', 'runner_callback', 'reap_callback', 'return_callback', 'clear'}
selected = [copy.deepcopy(n) for n in B.body if isinstance(n, ast.Assign) or
    isinstance(n, ast.FunctionDef) and n.name in NAMES]
namespace = {'Path': Path, '__file__': str(GUARD)}
module = ast.Module(body=[ast.ClassDef(name='BlockedPipeWitness_B', bases=[], keywords=[],
    body=selected, decorator_list=[])], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), '<extracted-B-predicates>', 'exec'), namespace)
C = namespace['BlockedPipeWitness_B']
checks = []
def check(label, predicate):
    assert predicate, label
    checks.append(label)
    assert len(checks) <= 64

def refused(label, call):
    try: call()
    except (ValueError, AttributeError): checks.append(label)
    else: raise AssertionError(label)

path = ROOT/'tests/service/test_recovery.py'
data = path.read_bytes()
code = C.nested(compile(data, str(path), 'exec', dont_inherit=True), C.name)
f = FunctionType(code, {})  # Carrier only: never called.
config = SimpleNamespace(getoption=lambda key: 'plain')
node = SimpleNamespace(config=config)
target = C.target_source(f, node)
facts = dict(key=C.fact_key(f, node, data), config=config, execute_code=code,
    outer_projection=C.code_projection(code), target=target)
check('plain-independent-baseline', C.target_source(f, node, facts) is target)
check('exact-byte-key', facts['key'][0] == data and facts['key'][1] == hashlib.sha256(data).digest())
check('changed-byte-key', C.fact_key(f, node, data+b'\n') != facts['key'])
refused('foreign-config', lambda: C.target_source(f, SimpleNamespace(config=SimpleNamespace(getoption=config.getoption)), facts))
config.getoption = lambda key: 'unknown'
refused('changed-mode', lambda: C.target_source(f, node, facts))
config.getoption = lambda key: 'plain'
f.__code__ = code.replace(co_qualname=code.co_qualname+'.changed')
refused('changed-code', lambda: C.target_source(f, node, facts))
f.__code__ = code.replace()
check('equal-code-distinct-identity-prerequisite', f.__code__ == code and f.__code__ is not code)
refused('equal-code-identity-refusal', lambda: C.target_source(f, node, facts))
f.__code__ = code.replace(co_filename=str(path.parent/'foreign.py'))
refused('changed-path', lambda: C.target_source(f, node, facts))
f.__code__ = code
# Source read substitution affects only this extracted predicate's Path binding.
class ChangedPath:
    def __new__(cls, value):
        class Proxy:
            def __init__(self, p): self.p = Path(p)
            def __truediv__(self, name): return Proxy(self.p/name)
            @property
            def parent(self): return Proxy(self.p.parent)
            def resolve(self): return self.p.resolve()
            def read_bytes(self): return self.p.read_bytes()+b'\n'
        return Proxy(value)
namespace['Path'] = ChangedPath
refused('fresh-source-digest-refusal', lambda: C.target_source(f, node, facts))
namespace['Path'] = Path
check('restored-source-baseline', C.target_source(f, node, facts) is target)
# Configuration hook key is read fresh without importing rewrite_asserts.
config.getoption = lambda key: 'rewrite'
config.getini = lambda key: False
key = C.fact_key(f, node, data)
config.getini = lambda key: True
check('rewrite-config-hook-key', C.fact_key(f, node, data) != key)
config.getoption = lambda key: 'plain'
# Exact installed cleanup function, with a synthetic holder: no install execution.
install = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'install')
clear = next(n for n in install.body if isinstance(n, ast.FunctionDef) and n.name == 'clear_prepared_B')
holder = {'facts': facts.copy()}; old = holder['facts']
cleanup_ns = {'prepared_B': holder}
exec(compile(ast.Module(body=[copy.deepcopy(clear)], type_ignores=[]), '<extracted-clear>', 'exec'), cleanup_ns)
cleanup_ns['clear_prepared_B']()
check('scope-clear', holder == {} and old == {})
cleanup_ns['clear_prepared_B']()
check('idempotent-finalization', holder == {})
refused('consecutive-old-facts-refused', lambda: C.target_source(f, node, old))
# Identity selector and required exception/return delivery with synthetic frames.
w = C(); w.valid = True
runner, reap, returns, foreign = (object() for _ in range(4))
w.runner, w.reap = runner, reap
seen = []
w.trace_B = lambda frame, event, arg: seen.append((frame.f_code, event))
w.local_callbacks = ((runner, w.runner_callback), (reap, w.reap_callback), (returns, w.return_callback))
for capability in (runner, reap, returns):
    frame = SimpleNamespace(f_code=capability, f_trace_lines=True)
    callback = w.select_B(frame, 'call', None)
    check('selector-lines-'+str(len(checks)), frame.f_trace_lines is (capability in (runner, reap)))
    for event in ('exception', 'return'):
        callback = callback(frame, event, None)
    check('selector-events-'+str(len(checks)), seen[-2:] == [(capability, 'exception'), (capability, 'return')])
check('foreign-selector', w.select_B(SimpleNamespace(f_code=foreign), 'call', None) is None)
w.valid = False
check('invalid-selector', w.select_B(SimpleNamespace(f_code=runner), 'call', None) is None)
# Extract only the reap event branch and run synthetic opcode/line transitions.
trace = next(n for n in B.body if isinstance(n, ast.FunctionDef) and n.name == 'trace_B')
branch = next(n for n in ast.walk(trace) if isinstance(n, ast.If) and ast.unparse(n.test) == 'code is self.reap')
program = compile(ast.Module(body=copy.deepcopy(branch.body), type_ignores=[]), '<extracted-kill-events>', 'exec')
kill = object(); os_fake = SimpleNamespace(kill=kill); signal_fake = SimpleNamespace(SIGKILL=9)
w.valid = True; w.kill = kill; w.child = 123; w.kill_sites = {20, 22}; w.kill_completion = 24
w.kill_lines = {593}; w.reap_sites = {661}; w.kill_pending = w.killed = False
parent = SimpleNamespace(f_code=runner, f_lasti=661, f_locals={'pid': 123})
frame = SimpleNamespace(f_lineno=581, f_lasti=0, f_trace_opcodes=False, f_back=parent,
    f_globals={'os': os_fake}, f_locals={'pid':123, 'signal':signal_fake})
def event(kind, offset=0, line=None):
    frame.f_lasti = offset
    if line is not None: frame.f_lineno = line
    exec(program, {'self':w, 'frame':frame, 'event':kind, 'os':os_fake, 'signal':signal_fake})
event('line', line=581)
check('wait-opcodes-off', not frame.f_trace_opcodes)
event('line', line=593); event('opcode', 20)
check('kill-call-not-completion', w.kill_pending and not w.killed)
event('exception')
check('kill-exception-invalidates-pending', not w.kill_pending and not frame.f_trace_opcodes and not w.killed)
event('line', line=593); event('opcode',20); event('opcode',22)
assert w.kill_pending and not w.killed
event('opcode',24)
check('kill-normal-completion', w.killed and not w.kill_pending and not frame.f_trace_opcodes)
w.killed = False
event('line', line=593); event('opcode',20); event('line', line=594)
check('line-invalidates-pending', not w.killed and not w.kill_pending and not frame.f_trace_opcodes)
os_fake.kill = object()
event('line', line=593); event('opcode',20); event('opcode',24)
check('changed-kill-refusal', not w.killed)
w.clear()
check('callback-cycle-cleared', not w.__dict__)
# Restoration ordering remains static: no real settrace, fixture or guard runs.
text = ast.unparse(install)
check('prior-trace-restoration-static', 'sys.settrace(trace_P)' in text and text.index('sys.settrace(trace_P)') < text.index('blocked_B.clear()'))
check('setup-clear-static', 'except BaseException:\n        clear_prepared_B()\n        raise' in text)
print('HOST-overhead PASS', len(checks), 'extracted/synthetic checks; full setup/fixture/trace restoration PENDING')
