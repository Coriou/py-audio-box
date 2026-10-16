"""Execute one pinned upstream export function with synthetic media and mocked DB/I/O.
This is a function-level compatibility probe, not an installed Voicebox test.
"""
import ast, asyncio, hashlib, json, tempfile, types
from pathlib import Path
import numpy as np
import soundfile as sf
source=Path('/upstream/stories.py')
tree=ast.parse(source.read_text())
function=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='export_story_audio')
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[])
class Table:
    id='id';generation_id='generation_id';story_id='story_id';start_time_ms='start_time_ms'
class Query:
    def __init__(self,pairs):self.pairs=pairs
    def filter_by(self,*a,**kw):return self
    def join(self,*a,**kw):return self
    def filter(self,*a,**kw):return self
    def order_by(self,*a,**kw):return self
    def first(self):return object()
    def all(self):return self.pairs
class DB:
    def __init__(self,pairs):self.pairs=pairs
    def query(self,*a,**kw):return Query(self.pairs)
root=Path(tempfile.mkdtemp(prefix='voicebox-probe-'))
valid=root/'present.wav';valid.write_bytes(b'synthetic-path-marker')
def save_audio(audio,path,rate): sf.write(path,audio,rate,subtype='FLOAT')
def make_tempfile(suffix):
    f=tempfile.NamedTemporaryFile(suffix=suffix,dir=root,delete=False);f.close();return Path(f.name)
ns=dict(DBStory=Table,DBStoryItem=Table,DBGeneration=Table,np=np,Path=Path,config=types.SimpleNamespace(resolve_storage_path=lambda p:Path(p)),load_audio=lambda path,sample_rate:(np.full(24000,0.75,dtype=np.float32),sample_rate),save_audio=save_audio,_make_tempfile=make_tempfile)
exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
def pair(path,start=0,volume=1):
    return (types.SimpleNamespace(version_id=None,trim_start_ms=0,trim_end_ms=0,volume=volume,start_time_ms=start),types.SimpleNamespace(audio_path=str(path),duration=1.0,text='Synthetic fixture'))
async def run(pairs):
    data=await ns['export_story_audio']('synthetic',DB(pairs));
    if data is None:return dict(returned_audio=False)
    import io
    audio,rate=sf.read(io.BytesIO(data));return dict(returned_audio=True,samples=len(audio),rate=rate,peak=float(np.max(np.abs(audio))),mean=float(np.mean(audio)))
async def main():
    fixtures={
        'missing_source_alongside_valid':await run([pair(valid),pair(root/'missing.wav',2000)]),
        'two_overlaps_with_peak_above_one':await run([pair(valid),pair(valid)]),
        'zero_clip_volume':await run([pair(valid,volume=0)]),
        'nonzero_half_clip_volume':await run([pair(valid,volume=.5)])}
    assert fixtures['missing_source_alongside_valid']['returned_audio'] is True
    assert fixtures['missing_source_alongside_valid']['samples']==24000
    assert fixtures['two_overlaps_with_peak_above_one']['peak']==1.0
    assert fixtures['zero_clip_volume']['peak']==.75
    assert fixtures['nonzero_half_clip_volume']['peak']==.375
    receipt=dict(scope='Pinned upstream export function only, mocked DB/audio loader, synthetic PCM; no installed app/SQL/browser/real media',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),fixtures=fixtures,observations=['missing asset skipped with valid output','overlap peak automatically normalized','volume zero treated as default one','nonzero clip gain applied'])
    Path('/results/result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
asyncio.run(main())
