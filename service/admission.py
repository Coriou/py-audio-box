"""Private control policy; synthetic containment is never ML qualification."""
import copy
import wave
from contracts import validate as oracle
from service.core import encode
from service.resource_store import ResourceStore

DEFAULT_BUDGET=dict(slots=1,resident_models=1,ram_bytes=192*1024**2,vram_bytes=0,
                    pids=48,queue=32,disk_bytes=64*1024**2,free_bytes=16*1024**2)


def finite_budget(value):
    if set(value)!=set(DEFAULT_BUDGET):raise ValueError('closed_resource_budget')
    if any(type(v)!=int or v<0 or v>2**53-1 for v in value.values()):raise ValueError('finite_resource_budget')
    if any(value[k]==0 for k in value if k not in ('vram_bytes','free_bytes')):raise ValueError('positive_resource_budget')
    return copy.deepcopy(value)


def native_demand(context):
    """Pure historical reservation rule; no hardware freshness or store writes."""
    from worker.native_protocol import native_execution
    spec = native_execution(context)
    limits = context['plan']['resolved']['profile']['limits']
    if (4 if context['plan']['operation']=='voice_prepare' else 2)*spec.reference_bytes+3*min(spec.max_output_bytes,1048576)+96000 > limits['disk_bytes']:
        raise ValueError('native_control_disk_budget')
    return dict(configuration=oracle.canonical_hash(context['execution']),
                slots=1,ram_bytes=spec.memory_bytes,vram_bytes=0,pids=1,
                disk_bytes=limits['disk_bytes'],attempt_disk_bytes=limits['disk_bytes'])


class Admission:
    def __init__(self,core,staging,budget=None,*,private_spool=None):
        self.budget=finite_budget(DEFAULT_BUDGET if budget is None else budget)
        self.store=ResourceStore(core,self.budget,(core.root,staging) if private_spool is None else (core.root,staging,private_spool))
        self.core,self.staging=core,staging
        # Older work cannot be silently assigned invented budgets.
        if core.db.execute('SELECT 1 FROM jobs WHERE id NOT IN (SELECT id FROM resource_reservations) LIMIT 1').fetchone():
            self.unaccounted=True
        else:self.unaccounted=False

    def demand(self,context,profiles):
        plan=context['plan']
        oracle.admission(dict(operation=plan['operation'],input_sha256=oracle.plan_hash(plan)),context,profiles)
        limits=plan['resolved']['profile']['limits'];request=plan['request']
        if len(request.get('text',request.get('reference_text','')))>limits['text_chars'] or request.get('variants',1)>limits['variants']:raise ValueError('resource_input_limit')
        if len(encode(context).encode())>48000:raise ValueError('resource_plan_limit')
        from worker.runtime_protocol import execution
        spec = execution(context)
        if spec is not None:
            if profiles is None: raise ValueError('no_authority')
            selected = next(p for p in profiles if p['id']==plan['resolved']['profile']['id'])
            if selected['verification'] != 'measured' or not selected['qualification_receipt'] or 'synthetic' in selected['qualification_receipt'].lower():
                raise ValueError('real_qualification_required')
            if spec.engine == 'qwen':
                from worker.engines.qwen_execution import current_hardware, current_native_hardware, inventory_from_wire
                (current_native_hardware if spec.device == 'mps' else current_hardware)(spec, inventory_from_wire(context['execution']['hardware']))
                if spec.device == 'mps':
                    from worker.native_protocol import native_binding
                    native_binding(context)
                if spec.device.startswith('cuda:') and limits['vram_bytes'] <= 0: raise ValueError('cuda_memory_required')
            if spec.engine == 'qwen' and spec.device == 'mps':
                return native_demand(context)
            return dict(configuration=oracle.canonical_hash(context['execution']),
                        slots=1,ram_bytes=spec.memory_bytes,vram_bytes=limits['vram_bytes'] if spec.engine == 'qwen' and spec.device.startswith('cuda:') else 0,pids=1 if spec.engine == 'qwen' and spec.device == 'mps' else 48,
                        disk_bytes=limits['disk_bytes'],attempt_disk_bytes=limits['disk_bytes'])
        # Closed installed runtime capability, not a general engine/profile adapter.
        if plan['resolved']['profile']['id']!='cpu-synthetic':
            raise ValueError('resource_runtime_unsupported')
        operation = plan['operation']
        required = oracle.required_asset_ids(plan)
        if set(required) != {a['id'] for a in plan['assets']}:
            raise ValueError('asset_dependencies_mismatch')
        if operation == 'voice_prepare':
            if required != [request['asset_id']]: raise ValueError('resource_runtime_unsupported')
        elif operation == 'speech_generate':
            if request['variants'] != 1 or request['reuse'] != 'never':
                raise ValueError('resource_runtime_unsupported')
        else:
            raise ValueError('resource_runtime_unsupported')
        # Core verifies owned exact managed media before reserving and allocation.
        if limits['ram_bytes']<192*1024**2 or limits['expensive_slots']<1 or limits['resident_models']<1:
            raise ValueError('resource_containment_unsupported')
        if limits['decoded_ticks']<4800 or limits['upload_bytes']<9644 or limits['disk_bytes']<2*1024**2:
            raise ValueError('resource_output_limit')
        return dict(configuration=oracle.canonical_hash(plan['resolved']['identity']['effective']),
                    slots=1,ram_bytes=192*1024**2,vram_bytes=0,pids=48,
                    disk_bytes=2*1024**2,attempt_disk_bytes=2*1024**2)

    def reserve_tx(self,job,context,profiles):
        if self.unaccounted:raise ValueError('resource_reconstruction_required')
        d=self.demand(context,profiles)
        self.store.reserve_tx(job['id'],'inference',d)

    def _preflight(self):
        return self.store._preflight(self.check)

    def check(self, *, publication=False):
        if self.store._token is None:
            return self.store.observe(lambda:self.check(publication=publication))
        if getattr(self,'_checking',False):raise ValueError('resource_reentrant_check')
        self._checking=True
        try:return self._scan(publication=publication)
        finally:self._checking=False

    def _scan(self, *, publication=False):
        self.store.check_disk()
        for row in self.store.rows():
            if row['state']!='executing':continue
            job,context=self.core._load(row['id'])
            stage=self.staging/job['attempts'][-1]['staging_id']
            if stage.exists():
                size=sum(p.stat().st_size for p in stage.iterdir() if p.is_file())
                if size>row['demand']['attempt_disk_bytes']:raise ValueError('resource_stage_growth')
                from worker.runtime_protocol import execution
                spec=execution(context)
                max_bytes=spec.max_output_bytes if spec else 9644
                max_frames=spec.max_frames if spec else 4800
                rate=spec.sample_rate if spec else 48000
                files=[('output.wav',max_bytes,max_frames,rate)]
                if spec is not None and context['plan'].get('reference_roles'):
                    frames=spec.reference_frames if spec.engine=='qwen' else (spec.out_tick-spec.in_tick)*spec.reference_rate//48000
                    files=[('reference.wav',44+2*frames,frames,spec.reference_rate),('audition.wav',max_bytes,max_frames,rate)]
                for name,byte_limit,frame_limit,expected_rate in files:
                    output=stage/name
                    if output.exists() and output.stat().st_size>byte_limit:raise ValueError('resource_output_bytes')
                    if output.exists() and publication:
                        with wave.open(str(output),'rb') as wav:
                            if wav.getnframes()>frame_limit or wav.getframerate()!=expected_rate or wav.getnchannels()!=1 or wav.getsampwidth()!=2:
                                raise ValueError('resource_decoded_limit')
