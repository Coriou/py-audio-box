import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('tts_benchmark', ROOT / 'benchmarks/tts/run.py')
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_corpus_counts_and_critical_assertions():
    items = benchmark.load_corpus(ROOT / 'benchmarks/tts/corpus.json')
    assert sum(i['kind'] == 'sentence' and i['language'] == 'fr' for i in items) == 30
    assert sum(i['kind'] == 'sentence' and i['language'] == 'en' for i in items) == 15
    assert sum(i['kind'] == 'paragraph' for i in items) == 6
    assert sum(i['kind'] == 'long_script' for i in items) == 2


def test_mock_benchmark_records_success_error_and_blind_audio(tmp_path):
    pytest.importorskip('soundfile')
    app = tmp_path / 'mock.py'
    app.write_text('''import json,sys
import soundfile as sf
import numpy as np
args=sys.argv
text=args[args.index('--text')+1]
if 'budget' in text: raise SystemExit(7)
sf.write(args[args.index('--output')+1],np.zeros(2400),24000)
print(json.dumps({'metadata': {'seed_supported': False}, 'sample_rate':24000,'channels':1,'duration_sec':0.1}))
''')
    out = tmp_path / 'out'
    code = benchmark.main(['--backend', 'kokoro', '--assets', str(tmp_path), '--app', str(app),
        '--out', str(out), '--ids', 'fr-01', 'fr-03'])
    assert code == 1
    payload = json.loads((out / 'results.json').read_text())
    assert [r['status'] for r in payload['attempts']] == ['ok', 'error']
    assert payload['attempts'][1]['exit_code'] == 7
    assert payload['attempts'][0]['rtf_wall'] > 0
    assert (out / 'blind/001.wav').exists()
    assert 'kokoro' not in (out / 'listen.html').read_text()
    assert len((out / 'results.csv').read_text().splitlines()) == 3


@pytest.mark.skipif(os.name != 'posix', reason='process group contract is POSIX')
def test_timeout_stops_descendant_work(tmp_path):
    marker = tmp_path / 'leaked.txt'
    app = tmp_path / 'hang.py'
    app.write_text('''import subprocess,sys,time
subprocess.Popen([sys.executable,'-c',sys.argv[1]])
time.sleep(10)
''')
    child = f"import time;from pathlib import Path;time.sleep(0.7);Path({str(marker)!r}).write_text('leaked')"
    result = benchmark.execute([sys.executable, str(app), child], 0.15,
                               tmp_path / 'stdout', tmp_path / 'stderr')
    assert result['timed_out'] and result['wall_sec'] < 4
    time.sleep(0.8)
    assert not marker.exists()


def test_invalid_corpus_literal_rejected(tmp_path):
    path = tmp_path / 'corpus.json'
    path.write_text(json.dumps({'items':[dict(id='x', language='fr', text='bonjour', critical_literals=['absent'])]}))
    with pytest.raises(ValueError, match='critical literal'):
        benchmark.load_corpus(path)
