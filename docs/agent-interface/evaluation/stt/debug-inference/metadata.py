import json,os,platform,time,resource,pathlib,subprocess
import numpy as np
import ctranslate2
print(json.dumps({'platform':platform.platform(),'env':{k:v for k,v in os.environ.items() if k.startswith(('CT2','OMP','MKL','OPENBLAS','KMP'))},'ct2':ctranslate2.__version__,'supported':sorted(ctranslate2.get_supported_compute_types('cpu')),'cpu_count':os.cpu_count(),'affinity':sorted(os.sched_getaffinity(0))}),flush=True)
for f in ['/proc/cpuinfo','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpu.stat','/proc/pressure/cpu','/proc/pressure/memory']:
 print(f)
 try: print(pathlib.Path(f).read_text()[:1800])
 except OSError as e: print(type(e).__name__)
np.show_config();np.show_runtime()
a=np.random.default_rng(42).random((1024,1024),dtype=np.float32)
t=time.perf_counter();c=time.process_time()
for _ in range(20): b=a@a
print(json.dumps({'matmul_20x1024_wall':time.perf_counter()-t,'process_cpu_seconds':time.process_time()-c,'checksum':float(b.sum())}),flush=True)
p=pathlib.Path(ctranslate2.__file__).parent
print(subprocess.run(['ldd',str(next(p.glob('_ext*.so')))],capture_output=True,text=True).stdout)
from faster_whisper import WhisperModel
import inspect
print(inspect.getsource(WhisperModel.__init__))
