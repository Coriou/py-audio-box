"""Finite runtime identity producer; raw loader observations remain in the receipt."""
import sqlite3, _sqlite3, sys, json, re, importlib.metadata, pathlib, subprocess, shutil, hashlib
import importlib.util
spec = importlib.util.spec_from_file_location('p_one_identity', pathlib.Path(__file__).with_name('identity_semantics.py'))
I = importlib.util.module_from_spec(spec)
spec.loader.exec_module(I)
assert __debug__ and not sys.flags.optimize
assert sys.implementation.name == 'cpython' and sys.version_info[:3] == (3, 14, 8)
pins = re.findall(r'^([A-Za-z0-9_.-]+)==([^\s\\]+)', pathlib.Path('tests/release/requirements.lock').read_text(), re.M)
assert len(pins) == 42 and len({n for n, v in pins}) == 42
installed = [[n, v, importlib.metadata.version(n)] for n, v in pins]
assert all(v == actual for n, v, actual in installed)
with sqlite3.connect(':memory:') as db:
    source_id = db.execute('select sqlite_source_id()').fetchone()[0]
    options = db.execute('pragma compile_options').fetchall()
assert sqlite3.sqlite_version == '3.40.1'
assert source_id == '2022-12-28 14:03:47 df5c253c0b3dd24916e4ec7cf77d3db5294cc9fd45ae7b9c5e82ad8197f3alt1'
commands = {}
def checked(name, argv):
    result = subprocess.run(argv, capture_output=True, check=False)
    if result.returncode != 0 or result.stderr:
        print(json.dumps(dict(command=name, returncode=result.returncode, stdout_hex=result.stdout.hex(), stderr_hex=result.stderr.hex())), file=sys.stderr)
    commands[name] = dict(returncode=result.returncode, stderr=result.stderr.decode('ascii'))
    I.require(result.returncode == 0 and not result.stderr, name + ' command')
    try:
        return result.stdout.decode('ascii')
    except UnicodeDecodeError:
        print(json.dumps(dict(command=name, stdout_hex=result.stdout.hex())), file=sys.stderr)
        raise

raw_ldd = checked('ldd', ['ldd', _sqlite3.__file__])
try:
    ldd = I.parse_ldd(raw_ldd.encode('ascii'))
except Exception:
    print(json.dumps(dict(raw_ldd=raw_ldd)), file=sys.stderr)
    raise
dpkg = checked('dpkg', ['dpkg-query', '-W', '-f=${Package} ${Version} ${Architecture}\n', 'libsqlite3-0'])
assert dpkg == 'libsqlite3-0 3.40.1-2+deb12u2 amd64\n'
lib = pathlib.Path('/usr/lib/x86_64-linux-gnu/libsqlite3.so.0').resolve()
digest = hashlib.sha256(lib.read_bytes()).hexdigest()
assert digest == '2e6eef9a727f081f0d453b4e5e6cbd8b9ef8b6f86cbf7681cbad444d3b0b55c8'
assert any(r[0] == 'library' and r[1] == 'libsqlite3.so.0' and pathlib.Path(r[2]).resolve() == lib for r in ldd)
assert shutil.disk_usage('/tmp').free >= 900 * 1024**2
receipt = dict(schema=1, raw_ldd=raw_ldd, semantic=dict(
    python=sys.version, implementation=sys.implementation.name, version=list(sys.version_info),
    optimization=sys.flags.optimize, assertions=__debug__, pins=installed,
    sqlite=sqlite3.sqlite_version, source_id=source_id, compile_options=options,
    extension=_sqlite3.__file__, ldd=ldd, library=dict(path=str(lib), sha256=digest),
    dpkg=dpkg, commands=commands))
raw = json.dumps(receipt, sort_keys=True)
I.semantic(raw)
print(raw)
