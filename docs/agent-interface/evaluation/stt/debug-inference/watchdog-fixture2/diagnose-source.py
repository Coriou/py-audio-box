"""Fixed public two-clip diagnostic. Stdout JSON stages survive a parent kill."""
import json,time,os,resource,threading,hashlib
from pathlib import Path
start=time.perf_counter();lock=threading.Lock();done=threading.Event()
def counters():
 out={'cpu_seconds':time.process_time(),'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 for f in ['cpu.stat','memory.current','memory.peak','memory.events']:
  try: out[f]=Path('/sys/fs/cgroup/'+f).read_text()
  except OSError: pass
 try: out['io']=Path('/proc/self/io').read_text()
 except OSError: pass
 return out
def emit(event,**k):
 with lock: print(json.dumps({'event':event,'wall':time.perf_counter()-start,**k}),flush=True)
def heartbeat():
 while not done.wait(3):
  active=[]
  for p in Path('/proc/self/task').glob('*/stat'):
   try:
    s=p.read_text();q=s[s.rindex(')')+2:].split();active.append({'tid':p.parent.name,'state':q[0],'user_ticks':int(q[11]),'sys_ticks':int(q[12]),'last_cpu':int(q[36]),'allowed_cpus':next(x for x in (p.parent/'status').read_text().splitlines() if x.startswith('Cpus_allowed_list'))})
   except OSError: pass
  emit('heartbeat',counters=counters(),threads=active)
threading.Thread(target=heartbeat,daemon=True).start()
from faster_whisper import WhisperModel
import ctranslate2,numpy as np,soundfile as sf
emit('environment',ct2=ctranslate2.__version__,env={k:v for k,v in os.environ.items() if k.startswith(('OMP','MKL','KMP','OPENBLAS','CT2'))},cpu_affinity=sorted(os.sched_getaffinity(0)),counters=counters())
threads=int(os.environ.get('DIAG_THREADS','4'))
emit('load_start')
t=time.perf_counter(); model=WhisperModel('/assets/models/small',device='cpu',compute_type='int8',cpu_threads=threads,num_workers=1,local_files_only=True)
emit('load_end',seconds=time.perf_counter()-t,effective_compute_type=str(model.model.compute_type),counters=counters())
def timed(name,fn,*a,**k):
 emit(name+'_start',counters=counters());t=time.perf_counter();cpu=time.process_time()
 result=fn(*a,**k)
 emit(name+'_end',seconds=time.perf_counter()-t,cpu_seconds=time.process_time()-cpu,counters=counters())
 return result
encode=model.encode
model.encode=lambda *a,**k:timed('encode',encode,*a,**k)
class Proxy:
 def __init__(self,obj):self.obj=obj
 def __getattr__(self,k):return getattr(self.obj,k)
 def generate(self,*a,**k):return timed('generate',self.obj.generate,*a,**k)
model.model=Proxy(model.model)
class Features:
 def __init__(self,obj):self.obj=obj
 def __getattr__(self,k):return getattr(self.obj,k)
 def __call__(self,*a,**k):return timed('features',self.obj,*a,**k)
model.feature_extractor=Features(model.feature_extractor)
manifest=json.loads(Path('/assets/manifest.json').read_text())
for sample_id in ['fr-1679','en-1914']:
 s=next(s for s in manifest['samples'] if s['id']==sample_id)
 emit('sample_start',id=s['id'],duration=s['duration'],sha256=hashlib.sha256(Path(s['file']).read_bytes()).hexdigest())
 t=time.perf_counter();signal,rate=sf.read(s['file'],dtype='float32');emit('audio_loaded',seconds=time.perf_counter()-t,samples=len(signal),rate=rate)
 t=time.perf_counter();cpu=time.process_time()
 segments,info=model.transcribe(signal,language=s['language'],task='transcribe',beam_size=5,vad_filter=False,word_timestamps=False,condition_on_previous_text=True,temperature=0,max_new_tokens=128)
 segments=list(segments)
 emit('sample_end',id=s['id'],seconds=time.perf_counter()-t,cpu_seconds=time.process_time()-cpu,text=''.join(s.text for s in segments).strip(),reference=s['text'],segments=[{'tokens':len(s.tokens),'start':s.start,'end':s.end} for s in segments],counters=counters())
done.set();emit('complete',counters=counters())
