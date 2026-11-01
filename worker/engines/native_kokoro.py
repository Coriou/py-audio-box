"""Explicit native CPU execution authority, distinct from Linux and Qwen/MPS.

The actual model call/verification/PCM encoding remains worker.engines.kokoro.
This wrapper carries native voice and deadline identity, never public paths.
"""
from dataclasses import dataclass
from worker.engines.execution import Execution
from worker.transport import Refused

VERSION = 'native-kokoro-execution-v1'
PROFILE = 'kokoro-native-cpu'
STARTUP_SECONDS = 30
REQUEST_SECONDS = 30
JOB_SECONDS = STARTUP_SECONDS + REQUEST_SECONDS
MEMORY_BYTES = 2 * 1024**3  # >3x the observed 604 MB sampled backend peak.


@dataclass(frozen=True)
class NativeKokoroExecution:
    model: Execution
    voice_id: str
    voice_revision: str

    def __post_init__(self):
        if type(self.model) is not Execution or self.model.engine != 'kokoro':
            raise Refused('native_cpu_execution')
        for value in (self.voice_id,self.voice_revision):
            if type(value) is not str or not 0<len(value)<=256: raise Refused('native_cpu_voice')
        if (self.model.language,self.model.preset,self.model.threads,self.model.memory_bytes) != ('en','af_heart',2,MEMORY_BYTES):
            raise Refused('native_cpu_preset')
        if self.model.max_text_bytes>256 or self.model.max_frames>24000*20 or self.model.max_output_bytes>1048576:
            raise Refused('native_cpu_limit')

    @classmethod
    def from_wire(cls,value):
        if type(value) is not dict or set(value)!={'version','model','voice_id','voice_revision'} or value['version']!=VERSION:
            raise Refused('native_cpu_execution')
        return cls(Execution.from_wire(value['model']),value['voice_id'],value['voice_revision'])

    def identity(self):
        return dict(version=VERSION,model=self.model.identity(),voice_id=self.voice_id,voice_revision=self.voice_revision)

    # Fixed preset-only native lifecycle surface. No Qwen task/dtype schema is
    # accepted or translated; the common owner only needs these read-only facts.
    @property
    def task(self): return 'preset'
    @property
    def reference_bytes(self): return 0
    @property
    def timeout_s(self): return REQUEST_SECONDS
    @property
    def device(self): return 'cpu'
    @property
    def dtype(self): return 'float32'
    @property
    def engine(self): return 'kokoro'
    @property
    def bundle_id(self): return self.model.bundle_id
    @property
    def manifest_sha256(self): return self.model.manifest_sha256
    @property
    def checkpoint(self): return self.model.checkpoint
    @property
    def revision(self): return self.model.revision
    @property
    def language(self): return self.model.language
    @property
    def preset(self): return self.model.preset
    @property
    def sample_rate(self): return self.model.sample_rate
    @property
    def channels(self): return self.model.channels
    @property
    def max_output_bytes(self): return self.model.max_output_bytes
    @property
    def memory_bytes(self): return self.model.memory_bytes
    @property
    def max_frames(self): return self.model.max_frames
    @property
    def threads(self): return self.model.threads
    @property
    def max_text_bytes(self): return self.model.max_text_bytes


def relation(context):
    """Bind only the new frozen native profile to a closed preset descriptor."""
    from worker.native_protocol import _closed
    from lib.spoken_text import speech_text
    _closed(context,('plan','bundles','execution'))
    value=context['execution'];_closed(value,('descriptor','bundles','native','hardware'))
    from worker.native_cpu_protocol import inventory
    inventory(value['hardware'])
    spec=NativeKokoroExecution.from_wire(value['descriptor'])
    plan=context['plan']; resolved=plan['resolved']; request=plan['request']
    profile=resolved['profile']; identity=resolved['identity']['effective']
    if value['bundles']!={spec.bundle_id:'/native-models/bundles/'+spec.bundle_id}:
        raise Refused('native_cpu_owned_mapping')
    if (profile['id'],profile['device'],profile['topology'])!=(PROFILE,'cpu','packaged_native'):
        raise Refused('native_cpu_topology')
    expected=dict(engine='kokoro',model=spec.checkpoint,bundle_sha256=spec.manifest_sha256,device='cpu',precision='float32')
    if (plan['operation']!='speech_generate' or request['fallback']!='forbid' or request['identity']!=identity or
            request['variants']!=1 or request['reuse']!='never' or plan['assets'] or identity!=expected):
        raise Refused('native_cpu_plan')
    versions=plan['voice_versions']
    if len(versions)!=1: raise Refused('native_cpu_voice')
    voice=versions[0]
    if (voice['id']!=spec.voice_revision or voice['voice_id']!=spec.voice_id or request['voice_version_id']!=voice['id'] or
            voice['origin']!='preset' or voice['reference_asset_ids'] or voice['identity']!=identity or
            voice['configuration']!=dict(mode='kokoro_preset',preset='af_heart',language='en') or
            resolved['settings']!=dict(kind='kokoro_preset',language='en')):
        raise Refused('native_cpu_voice')
    limits=profile['limits']
    if (limits['threads']!=2 or limits['ram_bytes']!=MEMORY_BYTES or limits['deadline_seconds']!=JOB_SECONDS or
            spec.max_frames*2>limits['decoded_ticks'] or spec.max_output_bytes>limits['upload_bytes'] or
            spec.max_output_bytes*2>limits['disk_bytes'] or len(speech_text(plan).encode())>spec.max_text_bytes):
        raise Refused('native_cpu_budget')
    return spec
