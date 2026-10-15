import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location('memory_benchmark', Path(__file__).resolve().parents[1] / 'benchmarks/tts/memory.py')
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)

FAKE_APP = r'''
import argparse,json,wave
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--jobs');p.add_argument('--report');a,_=p.parse_known_args()
records=[]
for job in json.loads(Path(a.jobs).read_text()):
    if job['text']=='FAIL':
        records.append({'status':'error'});continue
    out=Path(job['output'])
    if job['text']!='MISSING':
        with wave.open(str(out),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(b'\x01\x00'*240)
    out.with_suffix('.meta.json').write_text(json.dumps({'status':'ok','text_original':job['text']}))
    records.append({'status':'ok'})
Path(a.report).write_text(json.dumps(records))
print(json.dumps(records))
raise SystemExit(any(r['status']!='ok' for r in records))
'''


def fixture(tmp_path, texts):
    app = tmp_path / 'app.py'; app.write_text(FAKE_APP)
    assets = tmp_path / 'assets'; assets.mkdir()
    (assets / 'manifest.json').write_text(json.dumps({'backend':'kokoro','checkpoint':'synthetic-fp32'}))
    scripts = tmp_path / 'scripts.json'
    scripts.write_text(json.dumps({'items':[dict(id=str(n),language='fr',text=t) for n,t in enumerate(texts)]+[dict(id='en',language='en',text='Hello')]}))
    out = tmp_path / 'results'
    return ['--assets',str(assets),'--scripts',str(scripts),'--language','fr','--app',str(app),'--image-digest','synthetic','--out',str(out),'--timeout-sec','5'], out


def test_batch_preserves_exact_order_repeats_and_audio(tmp_path):
    args, out = fixture(tmp_path, [' court  ', 'long texte\nexact', 'court'])
    assert memory.main(args+['--repeat','2']) == 0
    report = json.loads((out/'results.json').read_text())
    assert len(report['sessions']) == 2
    assert [a['text'] for a in report['attempts']] == [' court  ', 'long texte\nexact', 'court']*2
    assert all(a['frames']==240 and a['sample_rate']==24000 and a['finite'] and len(a['decoded_pcm_sha256'])==64 for a in report['attempts'])
    assert report['environment']['image_digest']=='synthetic'
    assert all(Path(j['output']).is_absolute() for j in json.loads(next(out.glob('*.jobs.json')).read_text()))
    with pytest.raises(FileExistsError):
        memory.main(args)


def test_failed_and_missing_audio_attempts_retained(tmp_path):
    args, out = fixture(tmp_path, ['FAIL','MISSING','valid'])
    assert memory.main(args) == 1
    report = json.loads((out/'results.json').read_text())
    assert [a['status'] for a in report['attempts']] == ['error','error','ok']
    assert report['sessions'][0]['exit_code']==1
    assert list(out.glob('*.batch.report.json')) and list(out.glob('*.stderr.log'))


def test_smaps_sums_proportional_and_private_and_rejects_partial(tmp_path):
    for pid in ('10','11'):
        (tmp_path/pid).mkdir()
        (tmp_path/pid/'smaps_rollup').write_text('Rss: 100 kB\nPss: 60 kB\nPrivate_Clean: 10 kB\nPrivate_Dirty: 20 kB\n')
    assert memory.group_memory(10,tmp_path,lambda pid:10)==dict(rss=204800,pss=122880,uss=61440)
    (tmp_path/'11'/'smaps_rollup').write_text('Rss: 100 kB\nPss: 60 kB\n')
    assert memory.group_memory(10,tmp_path,lambda pid:10) is None
    (tmp_path/'11'/'smaps_rollup').unlink()
    assert memory.group_memory(10,tmp_path,lambda pid:10) is None


def test_missing_sample_does_not_mix_partial_totals_into_observed_peaks(tmp_path, monkeypatch):
    readings=iter([dict(rss=10,pss=8,uss=6),None])
    monkeypatch.setattr(memory,'group_memory',lambda group:next(readings,None))
    result=memory.execute([sys.executable,'-c','import time;time.sleep(.15)'],2,tmp_path/'stdout',tmp_path/'stderr')
    assert result['memory_samples']==1 and result['memory_missing_samples']>0
    assert not result['memory_complete']
    assert result['observed_complete_group_peak_bytes']==dict(rss=10,pss=8,uss=6)
    assert 'missing intervals may conceal higher peaks' in result['memory_scope']


def test_timeout_reaps_owned_process(tmp_path):
    result=memory.execute([sys.executable,'-c','import time;time.sleep(60)'],.1,tmp_path/'stdout',tmp_path/'stderr')
    assert result['timed_out'] and result['exit_code']!=0 and result['wall_sec']<5
