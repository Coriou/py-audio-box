"""Bounded unit CI transport, deterministic allocation and fail-closed receipts.

Only collect/run import pytest. r502 seeds are rounded heartbeat-interval file
proxies, not per-test durations. Frozen r510 timings are scheduling estimates,
not runtime upper bounds. All hashes cover canonical JSON bytes.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

BASE_SOURCE_HASH = 'b76e998b7ed8c335ed9294781e81fc725e8a30078a05250b1669adeaca92eca8'
VERSION = 1
CAP = 2 * 1024 * 1024
PREFIX = b'PAB_UNIT_V1:'
SHARDS = [f'{i:02}' for i in range(1, 7)]
SELECTION = ['tests', '-m', 'not integration and not redis_integration and not manual_b_probe',
             '--ignore=tests/local', '--ignore=tests/remote',
             '--ignore=tests/release/test_dispatcher.py',
             '--ignore=tests/release/test_packaged_lifecycle.py',
             '--ignore=tests/release/test_admission.py',
             '-p', 'no:cacheprovider', '-x', '-vv', '--tb=short']
SEEDS = dict(zip(['tests/service/test_child_wait_guard.py', 'tests/release/test_native_renewal.py',
                 'tests/release/test_native_history.py', 'tests/release/test_native_origin_voices.py',
                 'tests/service/test_export_retained_history.py', 'tests/service/test_dispatcher.py',
                 'tests/service/test_admission.py', 'tests/service/test_daemon_volumes.py'],
                [(176,167),(127,167),(68,141),(66,54),(44,24),(33,71),(28,80),(25,101)]))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write(path, value):
    data = canonical(value)
    assert len(data) <= CAP, 'metadata cap'
    Path(path).write_bytes(data)


def read(path):
    with Path(path).open('rb') as source:
        data = source.read(CAP+1)
    assert len(data) <= CAP, 'metadata cap'
    return json.loads(data)


TIMING_WEIGHTS = read(Path(__file__).with_name("ci_unit_timing_weights.json"))
RECOVERY = "tests/service/test_recovery.py"


def source_hash():
    excluded={b'.github/workflows/unit.yml',b'scripts/ci-unit.sh',b'scripts/ci_unit_shards.py'}
    entries=subprocess.check_output(['git','ls-tree','-rz','--full-tree','HEAD']).split(b'\0')
    return hashlib.sha256(b'\0'.join(e for e in entries if e and e.split(b'\t',1)[1] not in excluded)).hexdigest()


def validate_baseline(manifest):
    # Only the unchanged r503 source tree is count-pinned. Future source trees
    # use their own independent canonical collection, never a frozen 4481.
    if manifest['identity'].get('source_hash')==BASE_SOURCE_HASH:
        counts=(len(manifest['raw']),len(manifest['deselected']),len(manifest['rows']),len(manifest['skips']))
        assert counts==(4536,55,4481,1), 'r502 baseline collection drift requires explanation'


def plan(rows):
    nodes = [r['node'] for r in rows]
    assert len(nodes) == len(set(nodes)), 'duplicate collected node'
    files = {}
    for index, row in enumerate(rows):
        files.setdefault(row['file'], []).append((index, row))
    groups = []
    for file, entries in files.items():
        atoms = []
        for index, row in entries:
            if not atoms or atoms[-1][0] != row['atom']:
                atoms.append((row['atom'], []))
            atoms[-1][1].append((index, row))
        assert len({a[0] for a in atoms}) == len(atoms), 'noncontiguous atom'
        def atom_cost(key, atom):
            weight = TIMING_WEIGHTS['atom_weights'].get(key)
            if weight and weight['count'] == len(atom):
                return weight['cost_us']
            return 750000 * len(atom)

        packs = [entries] if file != RECOVERY and len(entries) <= 192 else []
        if not packs:
            pack = []
            pack_cost = 0
            for key, atom in atoms:
                cost = atom_cost(key, atom) if file == RECOVERY else 0
                if pack and (len(pack) + len(atom) > 192 or
                             (file == RECOVERY and pack_cost + cost > 180000000)):
                    packs.append(pack)
                    pack = []
                    pack_cost = 0
                pack += atom
                pack_cost += cost
            if pack:
                packs.append(pack)
        weight = TIMING_WEIGHTS['file_weights'].get(file)
        for pack in packs:
            count = len(pack)
            cost = 750000 * count
            if file == RECOVERY:
                keys = {row['atom'] for _, row in pack}
                cost = sum(atom_cost(key, atom) for key, atom in atoms if key in keys)
            elif weight and weight['count'] == len(entries):
                cost = (weight['cost_us'] * count + len(entries)-1)//len(entries)
            elif file in SEEDS and len(entries) == SEEDS[file][1]:
                cost = max(cost, (1250000 * SEEDS[file][0] * count + len(entries)-1)//len(entries))
            groups.append(dict(file=file, first=pack[0][0], count=count, cost=cost,
                               oversized=count > 192, indices=[i for i, _ in pack]))
    totals = {s: [0, 0, 0] for s in SHARDS}
    members = {s: [] for s in SHARDS}
    for group in sorted(groups, key=lambda g: (-g['cost'], -g['count'], g['file'], g['first'])):
        s = min(SHARDS, key=lambda s: (*totals[s], s))
        group['shard'] = s
        members[s] += [nodes[i] for i in group['indices']]
        totals[s] = [totals[s][0]+group['cost'], totals[s][1]+group['count'], totals[s][2]+1]
    for s in SHARDS:
        assigned = set(members[s])
        members[s] = [n for n in nodes if n in assigned]
        assert members[s], 'empty shard'
    assert Counter(n for ns in members.values() for n in ns) == Counter(nodes)
    return dict(groups=groups, assignments=members, totals=totals,
                file_counts={f: len(e) for f,e in files.items()},
                file_spread={f: sorted({g['shard'] for g in groups if g['file']==f}) for f in files})


class Record:
    """Incremental bounded receipt; a killed run leaves a detectably pending frame."""
    def __init__(self, name):
        self.name=name; self.seq=0; self.size=0; self.hash=hashlib.sha256()

    def send(self, data):
        assert self.size+len(data)<=CAP, 'metadata cap'
        self.size+=len(data); self.hash.update(data)
        for start in range(0,len(data),8192):
            frame=dict(name=self.name,seq=self.seq,data=base64.b64encode(data[start:start+8192]).decode())
            self.frame(frame); self.seq+=1

    def frame(self, value):
        data=b'\n'+PREFIX+canonical(value)+b'\n'
        while data:
            count=os.write(MACHINE_FD,data)
            assert count>0
            data=data[count:]

    def finish(self):
        self.frame(dict(name=self.name,end=True,size=self.size,sha=self.hash.hexdigest()))


def emit(name, value):
    record=Record(name); record.send(canonical(value)); record.finish()


def pytest_mode(args):
    global MACHINE_FD
    MACHINE_FD = os.dup(1)
    import pytest  # deliberately lazy: never imported by host modes
    from _pytest.subtests import SubtestReport

    class Audit:
        def __init__(self):
            self.raw=[]; self.deselected=[]; self.skips=[]; self.errors=[]
            self.event_count=0; self.subtests=Counter(); self.record=None; self.manifest=None; self.allocation=None

        def pytest_itemcollected(self, item):
            self.raw.append(item.nodeid)

        def pytest_deselected(self, items):
            self.deselected += [i.nodeid for i in items]

        def pytest_collectreport(self, report):
            if report.skipped:
                self.skips.append([report.nodeid, str(report.longrepr)])
            if report.failed:
                self.errors.append([report.nodeid, str(report.longrepr)])

        @pytest.hookimpl(wrapper=True, tryfirst=True)
        def pytest_collection_modifyitems(self, session, config, items):
            yield
            rows=[]
            for item in items:
                cls = item.getparent(pytest.Class)
                parent = cls if cls is not None else item.parent
                atom = cls.nodeid if cls is not None else parent.nodeid + '::' + (getattr(item,'originalname',None) or item.name)
                rows.append(dict(node=item.nodeid, file=item.path.relative_to(config.rootpath).as_posix(), atom=atom))
            assert len(self.raw)==len(set(self.raw)), 'duplicate raw collection'
            raw_index={n:i for i,n in enumerate(self.raw)}
            atoms={a:i for i,a in enumerate(dict.fromkeys(r['atom'] for r in rows))}
            compact=[dict(node=raw_index[r['node']],file=r['file'],atom=atoms[r['atom']]) for r in rows]
            self.node_index={r['node']:i for i,r in enumerate(rows)}
            self.manifest = dict(identity=json.loads(args.identity), version=VERSION,
                                 selection=SELECTION, raw=self.raw, deselected=self.deselected[:],
                                 skips=self.skips[:], errors=self.errors[:], rows=compact, atoms=list(atoms))
            validate_baseline(self.manifest)
            self.allocation=plan(rows)
            assert not self.errors, 'collection errors'
            if args.mode == 'run':
                assert digest(self.manifest)==args.manifest_hash, 'collection mismatch'
                assert digest(self.allocation)==args.plan_hash, 'plan mismatch'
                allowed=set(self.allocation['assignments'][args.shard])
                removed=[i for i in items if i.nodeid not in allowed]
                items[:]=[i for i in items if i.nodeid in allowed]
                config.hook.pytest_deselected(items=removed)
            emit('manifest', self.manifest)
            emit('plan', self.allocation)
            header=dict(identity=json.loads(args.identity),version=VERSION,shard=args.shard,
                        manifest_hash=digest(self.manifest),plan_hash=digest(self.allocation),
                        assigned=self.allocation['assignments'][args.shard] if args.mode=='run' else [])
            self.record=Record('receipt')
            assigned=header.pop('assigned'); self.header=header
            self.record.send(b'{"assigned":'+canonical(assigned)+b',"events":[')

        def event(self, value):
            self.record.send((b',' if self.event_count else b'')+canonical(value))
            self.event_count+=1

        def pytest_runtest_logstart(self, nodeid, location):
            self.event([self.node_index[nodeid], 'start'])

        def pytest_runtest_logreport(self, report):
            is_subtest=isinstance(report, SubtestReport)
            if is_subtest:
                assert report.when=='call', 'subtest phase'
                self.subtests[report.outcome]+=1
            self.event([self.node_index[report.nodeid], 'subtest' if is_subtest else report.when,
                        report.outcome, report.duration, hasattr(report, 'wasxfail')])

        def pytest_runtest_logfinish(self, nodeid, location):
            self.event([self.node_index[nodeid], 'finish'])

        def pytest_sessionfinish(self, session, exitstatus):
            if self.record is not None:
                self.header['exit']=int(exitstatus)
                self.header['subtests']=dict(self.subtests)
                self.record.send(b'],'+canonical(self.header)[1:])
                self.record.finish()

    # Set verbosity explicitly: repeated -q would only cancel SELECTION's -vv.
    # The audit emits every node; terminal trees/items only delay receipt EOF.
    code=pytest.main(SELECTION + (['--collect-only','--verbosity=-2','-ra'] if args.mode=='collect' else []), plugins=[Audit()])
    return int(code)


def retain(root, prefix, command):
    root=Path(root); root.mkdir(parents=True, exist_ok=True)
    child=subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
        os.killpg(child.pid, signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    sel=selectors.DefaultSelector(); tails={}; buffers={}; records={}; finished=set(); invalid=[]
    last=0.; stop_at=None
    def human(name, data):
        nonlocal last
        tails[name]=(tails[name]+data)[-262144:]
        (root/(prefix+'.'+name)).write_bytes(tails[name])
        latest=next((v for v in reversed(tails[name].splitlines()) if b'::' in v), b'')
        if latest:
            safe=''.join(chr(v) if 32<=v<127 else '?' for v in latest[-512:])
            (root/'latest-progress.txt').write_text('test progress: '+safe+'\n')
            if time.monotonic()-last>=15:
                print('test progress: '+safe, flush=True); last=time.monotonic()
    def line(name, value):
        if not value.startswith(PREFIX):
            human(name,value); return
        try:
            assert len(value)<=16384
            f=json.loads(value[len(PREFIX):]); key=f['name']
            assert key in {'manifest','plan','receipt'} and key not in finished
            seq,data=records.setdefault(key,(0,bytearray()))
            if f.get('end'):
                assert f['size']==len(data) and f['sha']==hashlib.sha256(data).hexdigest()
                json.loads(data); (root/(key+'.json')).write_bytes(data); finished.add(key)
            else:
                assert f['seq']==seq
                chunk=base64.b64decode(f['data'], validate=True)
                assert len(data)+len(chunk)<=CAP
                data.extend(chunk); records[key]=(seq+1,data)
        except Exception as exc:
            invalid[:] = (invalid + [type(exc).__name__])[:20]
    for name,stream in [('stdout',child.stdout),('stderr',child.stderr)]:
        sel.register(stream,selectors.EVENT_READ,name); tails[name]=b''; buffers[name]=b''
    try:
        while sel.get_map():
            if stopping:
                stop_at=stop_at or time.monotonic()
                if time.monotonic()-stop_at>2:
                    try:
                        os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            for key,_ in sel.select(.2):
                data=os.read(key.fd,65536); name=key.data
                if not data:
                    if buffers[name]:
                        line(name,buffers[name])
                    sel.unregister(key.fileobj); continue
                buffers[name]+=data
                while b'\n' in buffers[name]:
                    value,buffers[name]=buffers[name].split(b'\n',1); line(name,value+b'\n')
                if len(buffers[name])>16384:
                    if buffers[name].startswith(PREFIX):
                        invalid[:] = (invalid + ['oversize frame'])[:20]; buffers[name]=b''
                    else:
                        human(name,buffers[name][:-len(PREFIX)]); buffers[name]=buffers[name][-len(PREFIX):]
        try:
            code=child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            invalid.append('EOF without child exit')
            os.killpg(child.pid,signal.SIGKILL)
            code=child.wait()
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL); child.wait()
        (root/(prefix+'-follower-reaped.txt')).write_text(str(child.returncode)+'\n')
    if set(records)!=finished:
        invalid.append('incomplete frame')
        for name, (_,data) in records.items():
            if name not in finished:
                (root/(name+'.pending')).write_bytes(data)
    write(root/(prefix+'-transport.json'),dict(complete=not invalid and not stopping, errors=invalid[:20]))
    return 143 if stopping else 125 if invalid else code if code>=0 else 125


def verify_execution(receipt, assigned, full_nodes):
    assert receipt['assigned']==assigned and receipt['exit']==0, 'assignment/exit'
    events=receipt['events']; cursor=0; subtests=Counter()
    for name in assigned:
        node=full_nodes.index(name)
        assert events[cursor]==[node,'start'], 'missing/duplicate/start order'
        cursor+=1; phases=[]
        while cursor<len(events) and events[cursor][1]!='finish':
            event=events[cursor]
            assert len(event)==5, 'report shape'
            assert event[0]==node and event[1] in ('setup','call','teardown','subtest')
            assert event[2] in ('passed','skipped','failed') and event[2]!='failed'
            assert isinstance(event[3],(int,float)) and math.isfinite(event[3]) and event[3]>=0
            assert len(event)==5 and isinstance(event[4],bool)
            if event[1]=='subtest':
                assert [p[1] for p in phases]==['setup'] and phases[0][2]=='passed', 'subtest placement'
                subtests[event[2]]+=1
            else:
                phases.append(event)
            cursor+=1
        assert cursor<len(events) and events[cursor]==[node,'finish'], 'pending protocol'
        cursor+=1
        names=[p[1] for p in phases]
        assert names in (['setup','call','teardown'],['setup','teardown']), 'terminal phases'
        if len(phases)==2:
            assert phases[0][2]=='skipped', 'missing call'
        else:
            assert phases[0][2]=='passed'
        assert phases[-1][2] in ('passed','skipped'), 'teardown not complete'
    assert cursor==len(events), 'extra protocol'
    assert receipt.get('subtests', {})==dict(subtests), 'subtest counts'


def verify_directory(root):
    assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<=8*1024*1024, 'diagnostics cap'
    assert (root/'step-exit-code.txt').read_text().strip()=='0', 'runner failure'
    state=read(root/'state.json')
    assert state['Status']=='exited' and state['Running'] is False and state['Pid']==0 and state['OOMKilled'] is False and state['ExitCode']==0 and not state['Error'], 'terminal state'
    assert (root/'log-drain.txt').read_text().strip()=='drain=eof follower_exit=0', 'log drain'
    assert read(root/'container-transport.json')['complete'], 'transport'
    assert (root/'cleanup-ok.txt').read_text().strip()=='yes', 'cleanup'


class AggregationError(AssertionError):
    def __init__(self, summary):
        self.summary=summary
        super().__init__('; '.join(summary['errors']))


def aggregate(root, identity, dependencies, input_result='success'):
    root=Path(root); errors=[]; statuses={}; manifest=None; allocation=None; rows=[]
    if input_result!='success':
        errors.append('artifact download/checkout failure: '+input_result)
    if dependencies!={'prepare':'success','shards':'success'}:
        errors.append('dependency failure: '+str(dependencies))
    expected={'prepare', *('shard-'+s for s in SHARDS)}
    if not root.exists() or {p.name for p in root.iterdir()}!=expected:
        errors.append('missing or unexpected receipt sets')
    prep=root/'prepare'
    try:
        verify_directory(prep)
        manifest=read(prep/'manifest.json'); allocation=read(prep/'plan.json')
        assert manifest['identity']==identity and manifest['version']==VERSION, 'prepare identity'
        validate_baseline(manifest)
        assert manifest['selection']==SELECTION and not manifest['errors'], 'selection/errors'
        assert identity['version']==VERSION and identity['shards']==6 and identity['selection_hash']==digest(SELECTION), 'configuration identity'
        assert len(manifest['raw'])==len(set(manifest['raw'])), 'raw duplicates'
        rows=[dict(r,node=manifest['raw'][r['node']], atom=manifest['atoms'][r['atom']])
              for r in manifest['rows']]
        assert Counter(r['node'] for r in rows)+Counter(manifest['deselected'])==Counter(manifest['raw']), 'selection accounting'
        assert allocation==plan(rows), 'canonical plan mismatch'
        gate=read(prep/'receipt.json')
        assert gate['exit']==0 and gate['identity']==identity and gate['manifest_hash']==digest(manifest) and gate['plan_hash']==digest(allocation), 'collection gate receipt'
        assert gate['version']==VERSION and gate['events']==[] and gate['assigned']==[], 'collection gate execution'
        statuses['prepare']=dict(ok=True)
    except Exception as exc:
        errors.append('prepare: '+str(exc)[:300]); statuses['prepare']=dict(ok=False)
        allocation=None
    for shard in SHARDS:
        directory=root/('shard-'+shard); receipt=None
        assigned=allocation['assignments'][shard] if allocation else []
        status=dict(ok=False,assigned=len(assigned)); statuses[shard]=status
        try:
            verify_directory(directory)
            assert allocation is not None, 'no valid canonical gate'
            assert read(directory/'manifest.json')==manifest, 'collection mismatch'
            assert read(directory/'plan.json')==allocation, 'plan mismatch'
            receipt=read(directory/'receipt.json')
            assert receipt['identity']==identity and receipt['version']==VERSION and receipt['shard']==shard, 'shard identity'
            assert receipt['manifest_hash']==digest(manifest) and receipt['plan_hash']==digest(allocation), 'manifest/plan hashes'
            verify_execution(receipt,assigned,[r['node'] for r in rows])
            status['ok']=True
        except Exception as exc:
            status['error']=str(exc)[:300]; errors.append(shard+': '+status['error'])
        if not status['ok'] and assigned:
            # Diagnostic only: these finishes do not certify successful protocols.
            try:
                receipt=receipt or read(directory/'receipt.json')
                finished={e[0] for e in receipt['events'] if len(e)==2 and e[1]=='finish'}
            except Exception:
                finished=set()
            status['unfinished']=[r['node'] for i,r in enumerate(rows) if r['node'] in set(assigned) and i not in finished]
    summary=dict(ok=not errors,errors=errors,dependencies=dependencies,shards=statuses)
    if allocation:
        summary.update(selected=len(rows),manifest_hash=digest(manifest),plan_hash=digest(allocation),totals=allocation['totals'],
                       collection_counts=dict(raw=len(manifest['raw']),selected=len(rows),deselected=len(manifest['deselected']),skips=len(manifest['skips'])),
                       r502_reference=dict(raw=4536,selected=4481,deselected=55,skips=1))
    summary['phases']={}
    for directory in [prep]+[root/('shard-'+s) for s in SHARDS]:
        try:
            data=(directory/'phases.jsonl').read_bytes()
            assert len(data)<16384
            summary['phases'][directory.name]=[json.loads(line) for line in data.splitlines()]
        except Exception:
            pass
    if errors:
        raise AggregationError(summary)
    return summary


def stream_limit(limit, receipt=None):
    count=0
    while True:
        data=sys.stdin.buffer.read(65536)
        if not data: break
        count+=len(data)
        assert count<=limit, 'stream size limit'
        sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    if receipt:
        Path(receipt).write_text(str(count)+'\n')


def main():
    # Child options belong to the child, never to this CLI's parser.
    if len(sys.argv)>1 and sys.argv[1]=='retain':
        return retain(sys.argv[2],sys.argv[3],sys.argv[4:])
    if len(sys.argv)>1 and sys.argv[1]=='limit':
        stream_limit(int(sys.argv[2]),sys.argv[3] if len(sys.argv)>3 else None); return 0
    parser=argparse.ArgumentParser(); parser.add_argument('mode')
    parser.add_argument('--identity'); parser.add_argument('--shard',choices=SHARDS,default='01')
    parser.add_argument('--manifest-hash'); parser.add_argument('--plan-hash')
    parser.add_argument('rest',nargs='*'); args=parser.parse_args()
    if args.mode in ('collect','run'): return pytest_mode(args)
    if args.mode=='retain': return retain(args.rest[0],args.rest[1],args.rest[2:])
    if args.mode=='limit': stream_limit(int(args.rest[0])); return 0
    if args.mode=='aggregate':
        output=Path(args.rest[1]); output.mkdir(parents=True,exist_ok=True)
        try:
            summary=aggregate(args.rest[0],read(args.rest[2]),json.loads(args.rest[3]))
        except Exception as exc:
            write(output/'summary.json',getattr(exc,'summary',dict(ok=False,error=str(exc)[:2000]))); raise
        write(output/'summary.json',summary); return 0
    raise ValueError('unknown mode')


if __name__=='__main__':
    sys.exit(main())
