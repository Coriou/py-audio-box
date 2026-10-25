"""Closed PCM24 converter, called only inside the supervised attempt child.

Registration is trusted integration input, never an HTTP request. The supervisor
must independently bind its image digest and RO source mapping. Returning bytes
is a proposal; only the existing PID1 descendant-reap receipt permits publication.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
from contracts import validate as oracle
from worker.transport import Refused
from worker.engines.pocket import _open

MAX_SOURCE = 24*1024*1024
MAX_OUTPUT = 36*1024*1024
# Debian FFmpeg's resolved 214-library closure is about 10.5 KiB. Keep the
# registration finite while retaining every dependency (the outer plan/config
# limits remain unchanged).
MAX_REGISTRATION = 16*1024


def read_bounded(fd, limit):
    chunks=[]; total=0
    while total<=limit:
        data=os.read(fd,min(65536,limit+1-total))
        if not data: break
        chunks.append(data); total+=len(data)
    if total>limit: raise Refused('resource_input_limit')
    return b''.join(chunks)


def settings(value):
    oracle.validate('audio_export_request', value)
    loudness = value['loudness']
    if loudness is not None and (type(loudness) not in (int, float) or not math.isfinite(loudness)):
        raise Refused('invalid_request')
    if type(value['rate']) is not int or type(value['channels']) is not int:
        raise Refused('invalid_request')
    return copy.deepcopy(value)


def registration(value):
    oracle.validate('audio_converter_registration', value)
    if len(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8'))>MAX_REGISTRATION:
        raise Refused('resource_plan_limit')
    for path in [value['executable'], *value['runtime_closure']]:
        if not Path(path).is_absolute() or '..' in Path(path).parts or str(Path(path)) != path:
            raise Refused('converter_path')
    if len(set(value['runtime_closure'])) != len(value['runtime_closure']):
        raise Refused('converter_closure')
    return copy.deepcopy(value)


def pcm(raw, *, rate=None, channels=None, bits=None):
    """Complete bounded RIFF walker, including FFmpeg's PCM extensible header."""
    if len(raw) < 12 or raw[:4] != b'RIFF' or raw[8:12] != b'WAVE' or struct.unpack_from('<I',raw,4)[0]+8 != len(raw):
        raise Refused('invalid_media')
    offset=12; fmt=None; frames=None; count=0
    while offset < len(raw):
        count+=1
        if count>4096 or offset+8>len(raw): raise Refused('invalid_media')
        kind,size=struct.unpack_from('<4sI',raw,offset); start=offset+8; end=start+size+(size&1)
        if end>len(raw): raise Refused('invalid_media')
        if kind==b'fmt ':
            if fmt is not None or frames is not None or size not in (16,18,40): raise Refused('invalid_media')
            tag,ch,sr,br,align,width=struct.unpack_from('<HHIIHH',raw,start)
            if size==40:
                cb,valid,mask=struct.unpack_from('<HHI',raw,start+16)
                guid=raw[start+24:start+40]
                if tag!=65534 or cb!=22 or valid!=width or mask not in (0,4 if ch==1 else 3) or guid!=bytes.fromhex('0100000000001000800000aa00389b71'):
                    raise Refused('invalid_media')
            elif tag!=1 or (size==18 and raw[start+16:start+18]!=b'\0\0'):
                raise Refused('invalid_media')
            if ch not in (1,2) or not 8000<=sr<=96000 or width not in (16,24,32) or align!=ch*width//8 or br!=sr*align:
                raise Refused('invalid_media')
            fmt=(sr,ch,width,align)
        elif kind==b'data':
            if fmt is None or frames is not None or size%fmt[3]: raise Refused('invalid_media')
            frames=size//fmt[3]
        offset=end
    if offset!=len(raw) or fmt is None or frames is None or frames<=0: raise Refused('invalid_media')
    sr,ch,width,_=fmt
    if any(want is not None and want!=got for want,got in ((rate,sr),(channels,ch),(bits,width))): raise Refused('media_mismatch')
    return dict(frames=frames,sample_rate=sr,channels=ch,media_format='wav',media_subtype='PCM_'+str(width),sha256=hashlib.sha256(raw).hexdigest())


@dataclass(frozen=True)
class Export:
    context: dict
    engine: str = 'audio-export'

    @property
    def sample_rate(self): return self.context['plan']['request']['rate']
    @property
    def channels(self): return self.context['plan']['request']['channels']
    @property
    def max_frames(self):
        a=self.context['plan']['assets'][0]
        return (a['frames']*self.sample_rate+a['sample_rate']-1)//a['sample_rate']+1
    @property
    def max_output_bytes(self): return min(MAX_OUTPUT,4096+self.max_frames*self.channels*3)
    @property
    def memory_bytes(self): return self.context['plan']['resolved']['profile']['limits']['ram_bytes']
    @property
    def threads(self): return self.context['plan']['resolved']['profile']['limits']['threads']


def execution(context):
    if set(context)!={'plan','bundles','execution'} or context['bundles']!={}: raise Refused('closed_export_context')
    plan=context['plan']; oracle.validate('audio_export_plan',plan)
    r=plan['resolved']; request=settings(plan['request']); reg=registration(r['identity']['converter'])
    if context['execution']!={'converter':reg} or r['kind']!='audio_export' or r['identity']['profile_id']!=r['profile']['id']:
        raise Refused('export_binding')
    if r['settings']!=request or r['settings_sha256']!=oracle.canonical_hash(request) or plan['voice_versions'] or plan['project'] is not None:
        raise Refused('export_binding')
    assets=plan['assets']
    if len(assets)!=1 or assets[0]['id']!=request['asset_id']: raise Refused('asset_dependencies_mismatch')
    a=assets[0]; limits=r['profile']['limits']
    if a['media_format']!='wav' or a['media_subtype'] not in ('PCM_16','PCM_24','PCM_32') or a['channels'] not in (1,2) or not 8000<=a['sample_rate']<=96000:
        raise Refused('invalid_media')
    spec=Export(copy.deepcopy(context))
    source_bytes=4096+a['frames']*a['channels']*int(a['media_subtype'][4:])//8
    if not 0<a['frames']<=60*a['sample_rate'] or source_bytes>MAX_SOURCE or spec.max_output_bytes>limits['upload_bytes'] or a['frames']*48000>limits['decoded_ticks']*a['sample_rate'] or 2*(MAX_SOURCE+spec.max_output_bytes)>limits['disk_bytes'] or spec.memory_bytes<192*1024**2 or r['profile']['device']!='cpu' or r['profile']['topology']!='networkless_container' or r['profile']['compatibility'] or not r['profile']['id'].startswith('audio-export-'):
        raise Refused('resource_input_limit')
    return spec


def provenance(plan):
    a=plan['assets'][0]
    return dict(kind='audio_export',source_id=a['id'],source_sha256=a['sha256'],missing_fields=[])


def verify_output(context, raw, output):
    spec=execution(context)
    if len(raw)>spec.max_output_bytes: raise Refused('resource_output_bytes')
    measured=pcm(raw,rate=spec.sample_rate,channels=spec.channels,bits=24)
    if not max(1,spec.max_frames-2)<=measured['frames']<=spec.max_frames or output['provenance']!=provenance(context['plan']) or any(output[k]!=v for k,v in measured.items()):
        raise Refused('export_output_policy')
    return measured


def verify_converter(reg, image_digest, *, owner_uid=0):
    reg=registration(reg)
    if reg['worker_image_digest']!=image_digest: raise Refused('converter_image_mismatch')
    # No execution/version probe. Image identity binds interpreter/shared libraries.
    for path in reg['runtime_closure']:
        fd=_open(path,owner_uid=owner_uid); os.close(fd)
    fd=_open(reg['executable'],owner_uid=owner_uid)
    try:
        info=os.fstat(fd)
        if not info.st_mode&0o111 or info.st_mode&0o222 or info.st_size>128*1024**2: raise Refused('converter_identity')
        digest=hashlib.sha256()
        while True:
            chunk=os.read(fd,65536)
            if not chunk: break
            digest.update(chunk)
        if digest.hexdigest()!=reg['executable_sha256']: raise Refused('converter_identity')
        os.lseek(fd,0,0)
        return fd
    except BaseException:
        os.close(fd); raise


def execute(context, source_fd, stage, *, image_digest, run=None, converter_uid=0, timeout=25):
    """FFmpeg stays in the existing child's process tree; never starts a session.

    On any exception retain staging. Caller/PID1 must drain descendants, including
    after successful converter exit, before admitting publication or releasing capacity.
    """
    spec=execution(context); plan=context['plan']; reg=plan['resolved']['identity']['converter']; a=plan['assets'][0]
    if not 0<timeout<=25: raise Refused('expired_result')
    before=os.fstat(source_fd)
    if not stat.S_ISREG(before.st_mode) or before.st_size>MAX_SOURCE or before.st_nlink!=1: raise Refused('missing_media')
    os.lseek(source_fd,0,0); raw=read_bounded(source_fd,MAX_SOURCE); os.lseek(source_fd,0,0)
    measured=pcm(raw)
    if any(measured[k]!=a[k] for k in measured): raise Refused('missing_media')
    exe=verify_converter(reg,image_digest,owner_uid=converter_uid)
    directory=output=None
    try:
        stage=Path(stage)
        if any(p.is_symlink() for p in (stage,*stage.parents)): raise Refused('unsafe_path')
        directory=os.open(stage,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        info=os.fstat(directory)
        if info.st_uid not in (os.getuid(),10001) or stat.S_IMODE(info.st_mode)!=0o770:
            raise Refused('foreign_stage')
        output=os.open('output.wav',os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o640,dir_fd=directory)
        command=['/proc/self/fd/'+str(exe),'-nostdin','-v','error','-y','-protocol_whitelist','file,pipe','-threads',str(spec.threads),'-filter_threads',str(spec.threads),'-f','wav','-i','/proc/self/fd/'+str(source_fd),'-map','0:a:0','-vn','-sn','-dn','-map_metadata','-1']
        if plan['request']['loudness'] is not None: command+=['-af','loudnorm=I='+str(plan['request']['loudness'])+':TP=-1.0:LRA=11']
        command+=['-ar',str(spec.sample_rate),'-ac',str(spec.channels),'-c:a','pcm_s24le','-fs',str(spec.max_output_bytes),'-f','wav','/proc/self/fd/'+str(output)]
        (subprocess.run if run is None else run)(command,check=True,timeout=min(25,timeout),pass_fds=(exe,source_fd,output),close_fds=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env={'PATH':'/usr/bin:/bin','LANG':'C','OMP_NUM_THREADS':str(spec.threads)})
        after=os.fstat(source_fd)
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns): raise Refused('missing_media')
        os.fsync(output); os.lseek(output,0,0); data=read_bounded(output,spec.max_output_bytes)
        result=pcm(data,rate=spec.sample_rate,channels=spec.channels,bits=24)
        if len(data)>spec.max_output_bytes or not max(1,spec.max_frames-2)<=result['frames']<=spec.max_frames: raise Refused('resource_output_bytes')
        return data,result
    finally:
        if output is not None: os.close(output)
        if directory is not None: os.close(directory)
        os.close(exe)
