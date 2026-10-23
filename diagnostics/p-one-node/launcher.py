import os,json,subprocess,time,hashlib,tarfile,shutil,re,traceback,resource
from pathlib import Path,PurePosixPath
os.umask(0o077)
R=Path('/tmp/pab-p-one-node-launcher-20261023');R.mkdir(mode=0o700);D=Path(__file__).resolve().parent;C=Path(__file__).resolve().parents[2];lane=Path('/tmp/pab-worker-w1-docker-lane.lock');pin=json.loads((D/'source-manifest.json').read_text())['commit'];base='sha256:69bd9216869374d87fa14702ef380b4124a12f03f5f88a2681957ea81ee274cb';name='pab-p-one-node-launcher-20261023';tag='pab-p-one-node-launcher-source:20261023';manifest=json.loads((D/'source-manifest.json').read_text())['files'];owner=dict(pid=os.getpid(),token=os.urandom(24).hex(),pin=pin,evidence=str(R));deadline=time.monotonic()+1800;limits={str(k):resource.getrlimit(k) for k in (resource.RLIMIT_NOFILE,resource.RLIMIT_FSIZE,resource.RLIMIT_CORE)}
def call(label,args,timeout=60):
 assert time.monotonic()<deadline,'STOP outer deadline';timeout=min(timeout,deadline-time.monotonic());t=time.monotonic();rc=None;timed=False
 with (R/(label+'.stdout')).open('wb') as out,(R/(label+'.stderr')).open('wb') as err:
  try: rc=subprocess.run(args,stdout=out,stderr=err,timeout=timeout).returncode
  except subprocess.TimeoutExpired:timed=True
 (R/(label+'.json')).write_text(json.dumps(dict(argv=args,exit=rc,timeout=timed,seconds=time.monotonic()-t),indent=2));return rc

def hostcheck(stage):
 assert subprocess.run(['git','-C',str(C),'merge-base','--is-ancestor',pin,'HEAD'],check=False).returncode==0
 assert not subprocess.check_output(['git','-C',str(C),'diff','--name-only',pin,'--',*manifest])
 assert all(hashlib.sha256((C/p).read_bytes()).hexdigest()==v for p,v in manifest.items())
 assert shutil.disk_usage('/tmp').free>=6*1024**3
 (R/(stage+'-host.json')).write_text(json.dumps(dict(head=pin,files=len(manifest),all_match=True,free=shutil.disk_usage('/tmp').free,limits={str(k):resource.getrlimit(k) for k in (resource.RLIMIT_NOFILE,resource.RLIMIT_FSIZE,resource.RLIMIT_CORE)}),indent=2))
verify="from pathlib import Path;import json,hashlib;m=json.loads(Path('manifest.json').read_text());assert m;assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in m.items());print(len(m),'matches')"
live="import os,stat,json,hashlib;from pathlib import Path;out={};\nfor root,dirs,files in os.walk('/tmp',followlinks=False):\n for n in files:\n  p=Path(root)/n;s=p.lstat()\n  if stat.S_ISREG(s.st_mode):out['./'+str(p.relative_to('/tmp'))]=hashlib.sha256(p.read_bytes()).hexdigest()\nprint(json.dumps(out,sort_keys=True))"
def validate_records():
 with tarfile.open(R/'archive.stdout') as tf:
  raw=tf.extractfile('./pab-p-one-node-observer/records').read()
 assert len(raw)<=16384 and raw.endswith(b'\n')
 rows=[]
 for line in raw.splitlines():
  assert len(line)+1<=512 and re.fullmatch(rb'-?[0-9]+(?:,-?[0-9]+){46}',line)
  rows.append([int(x) for x in line.split(b',')])
 assert len(rows)==4 and sorted(r[2] for r in rows)==[1,2,3,4]
 child=next(r for r in rows if r[2]==1);wait=next(r for r in rows if r[2]==2);owner=wait[3]
 assert all(r[:2]==[1,1] and r[3]==owner for r in rows)
 assert child[4]==wait[6] and child[5]==owner and wait[7]==18944
 assert all(r[4]==owner for r in rows if r[2]!=1)
 assert all(r[11]==0 and r[12]==1 for r in rows)
 assert child[16]==1 and child[26]==1 and child[31]==1
 (R/'schema-binding.json').write_text(json.dumps(dict(structural=True,semantic_review_required=True,rows=rows)))
 # Actual caller/runner offsets and guard/report binding require independent receipt review.
 raise RuntimeError('STOP retained diagnostic requires independent offset/type/guard receipt review; no automatic cause acceptance')

def audit():
 assert call('archive',['docker','exec',name,'tar','--hard-dereference','-C','/tmp','-cf','-','.'],180)==0
 hashes={};inventory=[];total=0
 with tarfile.open(R/'archive.stdout') as tf:
  ms=tf.getmembers();assert len(ms)<200000 and len({m.name for m in ms})==len(ms)
  for m in ms:
   assert not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts
   assert m.isfile() or m.isdir() or m.issym()
   if m.isfile():
    assert m.uid==10001 and m.size<=64*1024**2
    total+=m.size;assert total<=1024**3
    hashes[m.name]=hashlib.sha256(tf.extractfile(m).read()).hexdigest()
   inventory.append(dict(name=m.name,size=m.size,uid=m.uid,mode=m.mode,type=m.type.decode(),link=m.linkname,sha256=hashes.get(m.name)))
 assert call('live-hashes',['docker','exec',name,'python','-c',live],180)==0
 assert hashes==json.loads((R/'live-hashes.stdout').read_text())
 (R/'inventory.json').write_text(json.dumps(inventory,indent=2));(R/'retention.json').write_text(json.dumps(dict(regular=len(hashes),members=len(ms),bytes=total,live_equal=True,sha256=hashlib.sha256((R/'archive.stdout').read_bytes()).hexdigest()),indent=2))
hostcheck('initial');(R/'source').mkdir(mode=0o700);
for path in manifest:
 src=C/path;assert src.is_file() and not src.is_symlink();dst=R/'source'/path;dst.parent.mkdir(parents=True,exist_ok=True,mode=0o700);dst.write_bytes(src.read_bytes());dst.chmod(0o600)
(R/'source'/'manifest.json').write_text(json.dumps(manifest));(R/'manifest.json').write_text(json.dumps(manifest));(R/'Dockerfile').write_text('FROM pab-p-one-node-launcher-owned-base:20261023\nWORKDIR /app\nCOPY source/ /app/\n');lane.mkdir(mode=0o700);(lane/'owner.json').write_text(json.dumps(owner));(R/'owner.json').write_text(json.dumps(owner));created=False;retained=False
try:
 assert call('base',['docker','image','inspect',base])==0
 assert json.loads((R/'base.stdout').read_text())[0]['Id']==base
 assert call('tag-absent',['docker','image','ls','-q','pab-p-one-node-launcher-owned-base:20261023'])==0 and not (R/'tag-absent.stdout').read_text().strip()
 assert call('base-tag',['docker','tag',base,'pab-p-one-node-launcher-owned-base:20261023'])==0
 assert call('tag-inspect',['docker','image','inspect','pab-p-one-node-launcher-owned-base:20261023'])==0
 assert json.loads((R/'tag-inspect.stdout').read_text())[0]['Id']==base
 assert call('derivative-absent',['docker','image','ls','-q',tag])==0 and not (R/'derivative-absent.stdout').read_text().strip()
 assert call('build',['docker','build','--pull=false','--network=none','-t',tag,str(R)],300)==0
 assert call('image',['docker','image','inspect',tag])==0
 image=json.loads((R/'image.stdout').read_text())[0]['Id']
 assert call('absent',['docker','ps','-aq','--filter','name=^/'+name+'$'])==0 and not (R/'absent.stdout').read_text().strip()
 args=['docker','create','--name',name,'--label','pab.owner='+owner['token'],'--cpus','2','--memory','2g','--pids-limit','256','--network','none','--read-only','--user','10001:10001','--cap-drop','ALL','--security-opt','no-new-privileges','--tmpfs','/tmp:rw,nosuid,nodev,size=1g,mode=1777','-e','PYTHONDONTWRITEBYTECODE=1','-e','TIKTOKEN_CACHE_DIR=/tokenizer-cache','-e','PAB_MAINTENANCE_WAIT_LOG=/tmp/waits.jsonl','-e','PYTHONPATH=/app/diagnostics/p-one-node',image,'python','-c','import signal,time;signal.signal(signal.SIGTERM,lambda *a:exit(0));time.sleep(1800)']
 assert call('create',args)==0;created=True
 assert call('start',['docker','start',name])==0
 prefix=['docker','exec','-w','/app',name]
 assert call('inspect',['docker','inspect',name])==0
 obj=json.loads((R/'inspect.stdout').read_text())[0];hc=obj['HostConfig'];cfg=obj['Config']
 assert hc['NanoCpus']==2000000000 and hc['Memory']==2147483648 and hc['PidsLimit']==256
 assert hc['NetworkMode']=='none' and hc['ReadonlyRootfs'] and cfg['User']=='10001:10001'
 assert obj['Id']==(R/'create.stdout').read_text().strip() and cfg['Labels']['pab.owner']==owner['token']
 assert hc['CapDrop']==['ALL'] and 'no-new-privileges' in hc['SecurityOpt']
 assert not hc['Binds'] and not hc['PortBindings'] and not hc['Devices']
 assert len(obj['Mounts'])==0 and list(hc['Tmpfs'])==['/tmp']

 assert call('admission',prefix+['python','-c',"import pathlib,json,shutil;v={a.split(':')[0]:int(a.split()[1])*1024 for a in pathlib.Path('/proc/meminfo').read_text().splitlines()};assert v['MemAvailable']>=4*1024**3;assert shutil.disk_usage('/').free>=4*1024**3;print(json.dumps(dict(memory_available=v['MemAvailable'],disk_free=shutil.disk_usage('/').free)))"])==0
 assert call('manifest-before',prefix+['python','-c',verify])==0
 identity=(D/'runtime-identity.py').read_text()
 identity+="\nassert sqlite3.sqlite_version=='3.40.1'\nassert hashlib.sha256(lib.read_bytes()).hexdigest()=='2e6eef9a727f081f0d453b4e5e6cbd8b9ef8b6f86cbf7681cbad444d3b0b55c8'\n"
 identity+="\nassert subprocess.check_output(['dpkg-query','-W','-f=${Version} ${Architecture}','libsqlite3-0'],text=True)=='3.40.1-2+deb12u2 amd64'\nwith sqlite3.connect(':memory:') as db: assert db.execute('select sqlite_source_id()').fetchone()[0]=='2022-12-28 14:03:47 df5c253c0b3dd24916e4ec7cf77d3db5294cc9fd45ae7b9c5e82ad8197f3alt1'\n"
 assert call('identity',prefix+['python','-c',identity])==0
 sentinel=owner['token'].encode()+b'\x00\xff\x80\r\n'
 assert call('sentinel-create',prefix+['python','-c',"from pathlib import Path;p=Path('/tmp/retention-sentinel');p.write_bytes("+repr(sentinel)+");p.chmod(0o600)"])==0
 assert call('sentinel',['docker','exec',name,'tar','--hard-dereference','-C','/tmp','-cf','-','retention-sentinel'])==0
 with tarfile.open(R/'sentinel.stdout') as tf:
  ms=tf.getmembers();assert len(ms)==1 and ms[0].uid==10001 and ms[0].mode==0o600 and tf.extractfile(ms[0]).read()==sentinel
 assert call('opcodes',prefix+['python','-c',"import dis,json,sys,hashlib,pathlib;print(json.dumps(dict(version=sys.version,opmap=dis.opmap,dis_sha256=hashlib.sha256(pathlib.Path(dis.__file__).read_bytes()).hexdigest())))"])==0
 hostcheck('before-test')
 assert call('tmp-capacity',prefix+['python','-c',"import shutil,json;d=shutil.disk_usage('/tmp');assert d.free>=900*1024**2;print(json.dumps(d._asdict()))"])==0
 # The one and only outer pytest command, exact predeclared selection/order.
 fullargv=prefix+['python','/app/diagnostics/p-one-node/entry.py','--one-node']
 assert time.monotonic()<deadline-1450
 (R/'execution-freeze.json').write_text(json.dumps(dict(argv=fullargv,env={'PYTHONPATH':'/app/diagnostics/p-one-node','PYTHONDONTWRITEBYTECODE':'1','PAB_MAINTENANCE_WAIT_LOG':'/tmp/waits.jsonl'},head=pin,harness='d5f493e63909d31e00680c81e8aefbcf3fcd5162',manifest_sha256=hashlib.sha256((R/'manifest.json').read_bytes()).hexdigest(),deadline_monotonic=deadline,test_timeout_seconds=1200),indent=2))
 rc=call('pytest',fullargv,1200)
 audit();retained=True
 assert call('manifest-after',prefix+['python','-c',verify])==0
 hostcheck('after-test')
 assert call('identity-after',prefix+['python','-c',identity])==0
 assert (R/'identity.stdout').read_bytes()==(R/'identity-after.stdout').read_bytes(),'STOP runtime identity drift'
 assert rc==1,('STOP unexpected pytest outcome',rc)
 (R/'expected-red.json').write_text(json.dumps(dict(exit=rc,diagnostic_only=True)))
 validate_records()
except BaseException:
 (R/'STOP.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
finally:
 if created:
  assert call('prestop',['docker','inspect',name])==0
  obj=json.loads((R/'prestop.stdout').read_text())[0];assert obj['Config']['Labels']['pab.owner']==owner['token'] and obj['Id']==(R/'create.stdout').read_text().strip() and obj['State']['Running']
  if retained:
   assert call('stop',['docker','stop','--time','10',name])==0
   assert call('stopped',['docker','inspect',name])==0
   state=json.loads((R/'stopped.stdout').read_text())[0]['State'];assert state['ExitCode']==0 and not state['OOMKilled'] and not state['Running']
   assert call('remove',['docker','rm',name])==0
   assert call('absence',['docker','ps','-aq','--filter','name=^/'+name+'$'])==0 and not (R/'absence.stdout').read_text().strip()
  else:(R/'LIVE-PROOF-HOLD.json').write_text(json.dumps(dict(container=name,owner=owner)))
 if not (R/'LIVE-PROOF-HOLD.json').exists():
  assert json.loads((lane/'owner.json').read_text())==owner
  (lane/'owner.json').unlink();lane.rmdir();(R/'lane-released.json').write_text(json.dumps(owner))
 hostcheck('final');assert limits=={str(k):resource.getrlimit(k) for k in (resource.RLIMIT_NOFILE,resource.RLIMIT_FSIZE,resource.RLIMIT_CORE)}
 print('DONE',flush=True)
