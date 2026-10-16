"""One fresh, hash-verified FR/EN auto-language probe of the portable build."""
import hashlib, json, subprocess, sys, importlib.util
from pathlib import Path

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[4]
ASSETS=ROOT/'work/stt-evaluation'
OUT=BASE/'pair-v1'
BUILD=BASE/'portable-v1'
IMAGE='py-audio-box:stt-cpp-avx2-toolchain-20261015'
NAME='pab-stt-debug-cpp-portable-pair-20261015'
if OUT.exists(): raise SystemExit('Refusing existing pair artifacts')
assert (BUILD/'build/bin/whisper-cli').is_file()
model=ASSETS/'models/whisper-cpp/ggml-tiny-q5_1.bin'
assert hashlib.sha256(model.read_bytes()).hexdigest()=='818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7'
manifest=json.loads((ASSETS/'manifest.json').read_text())
samples=[next(s for s in manifest['samples'] if s['id']==sid) for sid in ['fr-1679','en-1914']]
for sample in samples:
    actual=ASSETS/Path(sample['file']).relative_to('/assets')
    assert hashlib.sha256(actual.read_bytes()).hexdigest()==sample['sha256']
cmd=['docker','run','--name',NAME,'--network','none','--cpus','4','--memory','6g','--pids-limit','256','--mount',f'type=bind,src={ASSETS},dst=/assets,readonly','--mount',f'type=bind,src={BUILD},dst=/output,readonly','--mount',f'type=bind,src={OUT},dst=/results','--entrypoint','/usr/bin/timeout',IMAGE,'--signal=TERM','--kill-after=3s','85s','/output/build/bin/whisper-cli','--model','/assets/models/whisper-cpp/ggml-tiny-q5_1.bin','--no-gpu','--threads','4','--beam-size','5','--best-of','5','--language','auto','--max-context','0','--output-json']
for sample in samples: cmd+=['--file',sample['file'],'--output-file','/results/'+sample['id']]
code=subprocess.call([sys.executable,str(BASE.parent/'debug-inference/supervise.py'),'--name',NAME,'--seconds','90','--out',str(OUT),'--',*cmd])
(OUT/'inputs.json').write_text(json.dumps(samples,indent=2,ensure_ascii=False))
(OUT/'run-source.py').write_text(Path(__file__).read_text())
if code: raise SystemExit(code)
spec=importlib.util.spec_from_file_location('stt_metrics',BASE.parent/'benchmark.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
summary=[]
for sample in samples:
    result=json.loads((OUT/(sample['id']+'.json')).read_text())
    if not result.get('transcription'): raise RuntimeError('No transcription artifact: '+sample['id'])
    hypothesis=''.join(s['text'] for s in result['transcription']).strip()
    summary.append(dict(sample_id=sample['id'],reference=sample['text'],hypothesis=hypothesis,metrics=metrics.metrics(sample['text'],hypothesis),result=result.get('result')))
(OUT/'quality-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps(summary,ensure_ascii=False),flush=True)
