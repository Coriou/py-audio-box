"""Summarize only these immutable diagnostic receipts; no inference."""
import datetime,importlib.util,json,pathlib,re
p=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location('original_benchmark',p.parent/'benchmark.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
summary={'image':'sha256:fd86a754708e099477a8e624babe5e9880df365a100f18cdc784f55f917a038e','configs':{}}
for name in ['baseline','unpacked','spread','spread-repeat','turbo-spread']:
 d=p/name
 if not (d/'container.json').exists(): continue
 r=[json.loads(l) for l in (d/'stdout.log').read_text().splitlines()]
 sup=json.loads((d/'supervisor.json').read_text());container=json.loads((d/'container.json').read_text());state=container['State']
 seconds=(datetime.datetime.fromisoformat(state['FinishedAt'].replace('Z','+00:00'))-datetime.datetime.fromisoformat(state['StartedAt'].replace('Z','+00:00'))).total_seconds()
 stages=[{k:v for k,v in x.items() if k not in ['counters','env']} for x in r if x['event'] in ['load_end','features_end','encode_end','generate_end']]
 samples=[]
 for x in r:
  if x['event']=='sample_end':
   duration=next(s['duration'] for s in r if s['event']=='sample_start' and s['id']==x['id'])
   samples.append({k:v for k,v in x.items() if k!='counters'}|{'duration':duration,'rtf':x['seconds']/duration,'cpu_wall_ratio':x['cpu_seconds']/x['seconds'],'metrics':b.metrics(x['reference'],x['text'])})
 final=r[-1]
 speed=[int(x) for x in re.findall(r'CPU_Speed_Limit\s*=\s*(\d+)',(d/'host-load.log').read_text())]
 entry={'environment':next(x for x in r if x['event']=='environment'),'container_seconds':seconds,'state':state,'supervisor':sup,'stages':stages,'samples':samples,'last_counters':final.get('counters'),'host_cpu_speed_limit_samples':speed}
 if not samples:
  gs=next(x for x in r if x['event']=='generate_start');entry['generation_observed_lower_bound_seconds']=final['wall']-gs['wall'];entry['generation_observed_cpu_delta']=final['counters']['cpu_seconds']-gs['counters']['cpu_seconds']
 summary['configs'][name]=entry
summary['total_inference_container_seconds_including_load_cleanup']=sum(x['container_seconds'] for x in summary['configs'].values())
(p/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
for name,x in summary['configs'].items(): print(name,round(x['container_seconds'],3),[(s['id'],round(s['seconds'],3),round(s['rtf'],3)) for s in x['samples']],x['host_cpu_speed_limit_samples'])
print('totalcontainer',summary['total_inference_container_seconds_including_load_cleanup'])
