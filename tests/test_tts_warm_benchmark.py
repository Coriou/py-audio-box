import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

spec = importlib.util.spec_from_file_location('warm_benchmark', Path(__file__).resolve().parents[1] / 'benchmarks/tts/warm.py')
warm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warm)


FAKE_APP = r'''
import argparse,json,wave
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--jobs');p.add_argument('--report');p.add_argument('--text');p.add_argument('--output');a,_=p.parse_known_args()
def produce(job):
    if job['text']=='FAIL': return {'id':job['id'],'status':'error','error':'synthetic failure'}
    out=Path(job['output']);out.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(out),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(b'\x01\x00'*240)
    payload={'status':'ok','text_original':job['text'],'timings':{'generation_seconds':0.01}}
    out.with_suffix('.meta.json').write_text(json.dumps(payload));return dict(id=job['id'],**payload)
if a.command=='speak':
    result=produce({'id':'cold','text':a.text,'output':a.output});print(json.dumps(result));raise SystemExit(result['status']!='ok')
jobs_path=Path(a.jobs).resolve();jobs=json.loads(jobs_path.read_text())
for job in jobs:
    output=Path(job['output']);job['output']=str(output if output.is_absolute() else jobs_path.parent/output)
result={'attempts':[produce(j) for j in jobs]};result['status']='ok' if all(j['status']=='ok' for j in result['attempts']) else 'error'
Path(a.report).write_text(json.dumps(result));print(json.dumps(result));raise SystemExit(result['status']!='ok')
'''


def fixture(tmp_path, *, fail=False):
    app = tmp_path / 'app.py'; app.write_text(FAKE_APP)
    assets = tmp_path / 'assets'; assets.mkdir()
    (assets / 'manifest.json').write_text(json.dumps({'backend':'kokoro','checkpoint':'synthetic-fp32','files':[]}))
    corpus = tmp_path / 'corpus.json'
    corpus.write_text(json.dumps({'items':[{'id':'fr-01','language':'fr','text':'FAIL' if fail else 'Bonjour','critical_literals':[]},
                                          {'id':'en-01','language':'en','text':'Hello','critical_literals':[]}]}))
    out = tmp_path / 'report'
    args = ['--assets',str(assets),'--corpus',str(corpus),'--ids','fr-01','en-01',
            '--app',str(app),'--out',str(out),'--image-digest','synthetic-image','--timeout-sec','5']
    return args, out


def test_same_inputs_sequential_threads_numeric_reproducibility(tmp_path):
    args, out = fixture(tmp_path)
    assert warm.main(args) == 0
    report = json.loads((out / 'results.json').read_text())
    assert len(report['attempts']) == 12
    assert [g['threads'] for g in report['groups']] == [1,1,2,2,4,4]
    assert all(c['pcm_exact_equal'] and c['both_finite'] for c in report['comparisons'])
    assert all(g['process_wall_speedup'] > 0 for g in report['groups'])
    assert all(a['status']=='ok' and a['frames']==240 for a in report['attempts'])
    assert all('shared pages' in s['rss_scope'] for s in report['sessions'])


def test_failures_recorded_and_do_not_create_speedup(tmp_path):
    args, out = fixture(tmp_path, fail=True)
    assert warm.main(args+['--threads','1']) == 1
    report = json.loads((out / 'results.json').read_text())
    assert len(report['attempts']) == 4
    assert [a['status'] for a in report['attempts']] == ['error','error','ok','ok']
    assert report['groups'][0]['process_wall_speedup'] is None
    assert report['sessions'][1]['metadata']['attempts'][0]['error']=='synthetic failure'


def test_numerical_comparison_detects_changed_pcm(tmp_path):
    a,b=tmp_path/'a.wav',tmp_path/'b.wav'
    sf.write(a,np.array([0.0,0.5]),24000,subtype='PCM_24')
    sf.write(b,np.array([0.0,0.25]),24000,subtype='PCM_24')
    result=warm.compare_audio(a,b)
    assert result['same_rate_shape'] and result['both_finite']
    assert not result['pcm_exact_equal'] and result['pcm_max_abs_difference']==0.25


def test_process_deadline_terminates_owned_worker(tmp_path):
    result=warm.execute([sys.executable,'-c','import time; time.sleep(60)'],0.1,
                        tmp_path/'stdout',tmp_path/'stderr')
    assert result['timed_out'] and result['exit_code'] != 0
    assert result['wall_sec'] < 5


def test_relative_campaign_output_uses_batch_jobs_directory_resolution(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    args[args.index('--out')+1] = 'relative-results'
    assert warm.main(args+['--threads','1']) == 0
    out = tmp_path / 'relative-results'
    report = json.loads((out / 'results.json').read_text())
    assert len(report['attempts']) == 4
    assert all(attempt['status'] == 'ok' for attempt in report['attempts'])
    assert not (out / 'relative-results').exists()
    for jobs_path in out.glob('*.jobs.json'):
        for job in json.loads(jobs_path.read_text()):
            assert Path(job['output']).is_absolute()
            assert Path(job['output']).parent == out
