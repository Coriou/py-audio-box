import sqlite3,_sqlite3,sys,json,re,importlib.metadata,pathlib,subprocess,shutil
assert sys.version_info[:3]==(3,14,8)
pins=re.findall(r'^([A-Za-z0-9_.-]+)==([^\s\\]+)',pathlib.Path('tests/release/requirements.lock').read_text(),re.M)
assert len(pins)==42,len(pins)
assert all(importlib.metadata.version(n)==v for n,v in pins)
db=sqlite3.connect(':memory:')
print(json.dumps(dict(python=sys.version,sqlite=sqlite3.sqlite_version,source_id=db.execute('select sqlite_source_id()').fetchone(),compile_options=db.execute('pragma compile_options').fetchall(),extension=_sqlite3.__file__,pins=pins)))
db.close()
subprocess.run(['ldd',_sqlite3.__file__],check=True)
subprocess.run(['dpkg-query','-W','-f=${Package} ${Version} ${Architecture}\n','libsqlite3-0'],check=True)
lib=pathlib.Path('/usr/lib/x86_64-linux-gnu/libsqlite3.so.0').resolve()
import hashlib
print('linked-library',str(lib),hashlib.sha256(lib.read_bytes()).hexdigest())
assert shutil.disk_usage('/tmp').free>=900*1024**2
