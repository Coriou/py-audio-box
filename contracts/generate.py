"""Deterministic synthetic fixtures/media. Run: python -m contracts.generate."""
import base64
import hashlib
import json
import math
import struct
import wave
from pathlib import Path
ROOT=Path(__file__).parent/'fixtures/v1'
H='0'*64
from .scenarios import catalogue, identity, preset_version, transcript
IDENTITY=identity(catalogue()[0])
RESOLUTION=dict(requested=IDENTITY,effective=IDENTITY,fallback='forbid',reason='',profile_id='cpu-synthetic')

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def production(n,scale=False):
    assets=[]
    for name in ('impulse','sine','long','short'):
        assets.append(dict(media_subtype='PCM_16',media_format='wav',id=name,sha256=hashlib.sha256((ROOT/(name+'.wav')).read_bytes()).hexdigest(),frames=(2400 if name=='short' else 9600 if name=='long' else 4800),sample_rate=48000,channels=1,provenance=dict(kind='synthetic',source_sha256=hashlib.sha256((ROOT/(name+'.wav')).read_bytes()).hexdigest(),source_id=name,missing_fields=[])))
    voices=[dict(id=f'voice.{i}',aliases=[f'legacy_{i}'],version_ids=[f'vv{i}'],archived=False,revision=1) for i in range(3)]
    versions=[preset_version(i,assets[0]['provenance']) for i in range(3)]
    blocks=[];takes=[];clips=[]
    for i in range(n):
        text=f'  Ligne {i:03d} — café!\nExact text.  '
        blocks.append(dict(id=f'b{i}',speaker_id=f's{i%3}',text=text,selected_take_id=f't{i}',settings_sha256=H))
        takes.append(dict(id=f't{i}',block_id=f'b{i}',text=text,voice_version_id=f'vv{i%3}',settings_sha256=H,asset_id='impulse',identity=RESOLUTION))
        clips.append(dict(id=f'c{i}',track_id='speech',asset_id='impulse',take_id=f't{i}',start=i*4800,in_tick=0,out_tick=4800,gain=1,fade_in=0,fade_out=0,mute=False))
    # Imported bed overlaps speech. Scale has exactly two clips per block.
    for i in range(n if scale else 1):
        clips.append(dict(id=f'bed{i}',track_id='bed',asset_id='sine',take_id=None,start=i*4800+2400,in_tick=0,out_tick=4800,gain=0.25,fade_in=480,fade_out=480,mute=False))
    takes.append({**takes[0],'id':'candidate-long','asset_id':'long'})
    takes.append({**takes[0],'id':'candidate-short','asset_id':'short'})
    return dict(schema_version='1.3.0',id=f'production-{n}',revision=1,speakers=[dict(id=f's{i}',voice_version_id=f'vv{i}') for i in range(3)],blocks=blocks,takes=takes,tracks=[dict(id=x,mute=False,solo=False) for x in ('speech','bed')],clips=clips,assets=assets,voices=voices,voice_versions=versions,render=dict(sample_rate=48000,format='wav_pcm24',channels=2,channel_mapping='mono_duplicate_stereo_average',clipping='fail',include_stale=False,length_ticks=n*4800+2400))

def generate():
    ROOT.mkdir(parents=True,exist_ok=True)
    for name in ('impulse','sine','long','short'):
        frames=2400 if name=='short' else 9600 if name=='long' else 4800
        data=[(16384 if i==0 else 0) if name=='impulse' else round(8192*math.sin(2*math.pi*1000*i/48000)) for i in range(frames)]
        with wave.open(str(ROOT/(name+'.wav')),'wb') as f:
            f.setparams((1,2,48000,frames,'NONE','not compressed'));f.writeframes(struct.pack('<'+str(frames)+'h',*data))
    dump(ROOT/'production-30.json',production(30))
    dump(ROOT/'production-100.json',production(100,True))
    profiles=[]
    for device in ('cpu','cuda','metal'):
        profiles.append(dict(config_revision='config-1',compatibility=[],id=device+'-synthetic',device=device,topology='packaged_native' if device=='metal' else 'networkless_container',support='unqualified',installation='absent',configuration='valid',readiness='not_ready',verification='synthetic',checked_at='2026-10-16T00:00:00Z',qualification_receipt='',limits=dict(expensive_slots=1 if device=='cpu' else 2,resident_models=1,threads=2,queue=32,upload_bytes=16777216,decoded_ticks=48000*600,text_chars=10000,variants=4,disk_bytes=1073741824,deadline_seconds=600,ram_bytes=4294967296,vram_bytes=0 if device=='cpu' else 8589934592),next_action='Inventory host and qualify exact backend/model/precision before advertising support.'))
    dump(ROOT/'capabilities.json',dict(schema_version='1.3.0',profiles=profiles,operations=['discover','speech_generate'],engines=['qwen','kokoro','pocket-tts'],source_types=['upload','microphone','https','youtube'],checked_at='2026-10-16T00:00:00Z'))
    dump(ROOT/'negative.json',[
        dict(name='end-exclusive-empty',path=['clips',0,'out_tick'],value=0,error='invalid_trim'),
        dict(name='missing-media',path=['clips',0,'asset_id'],value='absent',error='missing_media'),
        dict(name='fade-overrun',path=['clips',0,'fade_in'],value=4801,error='invalid_fades'),
        dict(name='wrong-take-audio',path=['clips',0,'asset_id'],value='sine',error='take_asset_mismatch'),
        dict(name='wrong-selected-block',path=['blocks',0,'selected_take_id'],value='t1',error='selection_invalid'),
        dict(name='unknown-speaker',path=['blocks',0,'speaker_id'],value='absent',error='speaker_missing')])
    dump(ROOT/'expected.json',dict(version='1.3.0',tick_rate=48000,production_30=dict(blocks=30,speakers=3,clips=31,length_ticks=146400),production_100=dict(blocks=100,clips=200,length_ticks=482400),media=dict(frames=4800,impulse_first_sample=0.5,sine_frequency_hz=1000),cases=[dict(action='replace_text_b0',stale=['t0','candidate-long','candidate-short'],inference_starts=0,unchanged_clip_ids=['c0','c1']),dict(action='select_longer_take',following_clip_start_unchanged=4800,overlap_visible=True),dict(action='select_shorter_than_trim',error='invalid_trim',old_arrangement_preserved=True)]))
    schema=json.loads((ROOT.parents[1]/'v1/schema.json').read_text())['$defs']
    ops=json.loads((ROOT.parents[1]/'v1/operations.json').read_text())['operations']
    def example(spec):
        if '$ref' in spec:
            name=spec['$ref'].split('/')[-1]
            return production(30) if name=='project' else example(schema[name])
        if 'const' in spec:return spec['const']
        if 'enum' in spec:return spec['enum'][0]
        if 'anyOf' in spec:return example(spec['anyOf'][0])
        if spec.get('type')=='object':return {k:example(v) for k,v in spec['properties'].items()}
        if spec.get('type')=='array':return [example(spec['items']) for _ in range(spec.get('minItems',0))]
        if spec.get('type') in ('integer','number'):return max(1,spec.get('minimum',0))
        if spec.get('type')=='boolean':return False
        if spec.get('pattern')=='^[a-f0-9]{64}$':return H
        if spec.get('pattern','').startswith('^https'):return 'https://example.invalid/media.wav'
        return 'example'
    requests={op['id']:example(schema[op['request']]) for op in ops}
    # Fixture serialization only: public upload is an immutable raw WAV body.
    requests['asset_upload']={'$fixture_binary_base64':base64.b64encode((ROOT/'short.wav').read_bytes()).decode('ascii')}
    requests['voice_versions_list']=dict(cursor='',limit=1,voice_id='voice.0')
    dump(ROOT/'operation-requests.json',requests)
    for operation in ('voice_prepare','voice_design','speech_generate','project_render'):
        dump(ROOT/('transcript-'+operation+'.json'),transcript(operation,production(30),profiles,requests))
    for operation in ('voice_prepare','voice_design'):
        dump(ROOT/('transcript-'+operation+'-roles.json'),transcript(operation,production(30),profiles,requests,reference_roles=True))
    dump(ROOT/'job.json',dict(id='job1',operation='speech_generate',input_sha256=H,principal_id='principal1',idempotency_key='request1',state='running',generation=1,attempts=[dict(id='attempt1',generation=1,owner_id='worker1',reaped=False,staging_id='staging1')],artifact_ids=[],stage='loading',deadline='2026-10-16T00:10:00Z'))
    dump(ROOT/'artifact.json',dict(id='artifact1',job_id='job1',generation=1,asset_ids=['impulse'],input_sha256=H,project_revision=1,manifest_sha256=H,identity=RESOLUTION,attempt_id='attempt1',owner_id='worker1',staging_id='staging1'))
if __name__=='__main__':generate()
