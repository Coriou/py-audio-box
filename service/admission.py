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


class Admission:
    def __init__(self,core,staging,budget=None):
        self.budget=finite_budget(DEFAULT_BUDGET if budget is None else budget)
        self.store=ResourceStore(core,self.budget,(core.root,staging))
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
            selected = next(p for p in profiles if p['id']=='kokoro-cpu')
            if selected['verification'] != 'measured' or not selected['qualification_receipt'] or 'synthetic' in selected['qualification_receipt'].lower():
                raise ValueError('real_qualification_required')
            return dict(configuration=oracle.canonical_hash(context['execution']),
                        slots=1,ram_bytes=spec.memory_bytes,vram_bytes=0,pids=48,
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
                rate=24000 if spec else 48000
                output=stage/'output.wav'
                if output.exists() and output.stat().st_size>max_bytes:raise ValueError('resource_output_bytes')
                if output.exists() and publication:
                    with wave.open(str(output),'rb') as wav:
                        if wav.getnframes()>max_frames or wav.getframerate()!=rate or wav.getnchannels()!=1 or wav.getsampwidth()!=2:
                            raise ValueError('resource_decoded_limit')
