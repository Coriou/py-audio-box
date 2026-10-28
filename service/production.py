"""Trusted synthetic production domain; no HTTP, scheduler or engine readiness.

Contexts are operator-owned resolution snapshots, never public request payloads.
Candidate metadata is read exclusively from Core's committed publication.
"""
import copy
import json
from contracts import validate as oracle
from service.voice_store import _id
from worker.runtime_protocol import QWEN_CPU_SECONDS


class Production:
    def __init__(self, core):
        self.core = core

    def audio_export(self, principal, key, context, *, _admission=None, _profiles=None):
        from worker.audio_export import execution
        execution(context)
        return self.core.submit_execution(principal,key,context,_admission=_admission,
            _profiles=_profiles,_dispatch_settings=('audio_export',min(25,context['plan']['resolved']['profile']['limits']['deadline_seconds'])))

    def register_preset(self, principal, key, version, *, expected_revision):
        return self.core.voices.register(principal, key, version, expected_revision=expected_revision)

    def prepare(self, principal, key, context, *, _admission=None, _profiles=None, _dispatch_settings=None):
        context = copy.deepcopy(context)
        plan = context['plan']
        oracle.require(plan['operation'] == 'voice_prepare', 'operation_mismatch')
        from worker.runtime_protocol import execution, dispatch_mode
        if context.get('execution', {}).get('descriptor', {}).get('phase') == 'prepare' and context['execution']['descriptor'].get('engine') == 'pocket-tts' and plan['request']['reference_text'] != '':
            from worker.transport import Refused
            raise Refused('unsupported_parameter')
        asset_id = plan['request']['asset_id']
        self.core.voices._asset(principal, asset_id)
        plan['assets'] = [self.core.asset(asset_id)]
        token = oracle.canonical_hash([principal, key, plan['request']])
        reference = 'reference-' + token
        plan['resolved']['reuse_configuration']['reference_asset_id'] = reference
        audition_text=plan['request'].get('audition_text', plan.get('reference_roles',{}).get('audition_text','Synthetic fixture audition.'))
        if context.get('execution', {}).get('descriptor', {}).get('phase') == 'prepare':
            oracle.require(bool(plan['request'].get('audition_text', '').strip()), 'explicit_audition_required')
        plan['reference_roles'] = dict(version='reference-roles-v1', reference_kind='prepared_reference',
            reference_asset_id=reference, audition_asset_id='audition-'+token,
            audition_text=audition_text)
        spec = execution(context)
        if spec is not None:
            from worker.engines.pocket import _read
            from worker.engines.pocket_prepare import PocketPreparation, selected_reference
            from worker.engines.qwen_prepare import QwenPreparation
            oracle.require(type(spec) in (PocketPreparation, QwenPreparation), 'typed_preparation_required')
            if type(spec) is QwenPreparation:
                from worker.engines.qwen_prepare import selected_reference
            raw = _read(self.core.root/'assets'/spec.reference_id, spec.reference_bytes, spec.reference_sha256)
            selected_reference(spec, raw)
            _dispatch_settings = (dispatch_mode(spec), min(180 if spec.engine=='pocket-tts' else (QWEN_CPU_SECONDS if spec.engine=='qwen' and spec.device=='cpu' else 25), plan['resolved']['profile']['limits']['deadline_seconds']))
        return self.core.submit_execution(principal, key, context, _admission=_admission,
                                          _profiles=_profiles, _dispatch_settings=_dispatch_settings)

    def results(self, principal, job_id):
        job, context = self.core._load(job_id)
        oracle.require(job['principal_id'] == principal, 'forbidden')
        oracle.require(job['operation'] == 'voice_prepare' and job['state'] == 'succeeded', 'unpublished_candidate')
        row = self.core.db.execute('SELECT manifest,artifact FROM publications WHERE job_id=?', (job_id,)).fetchone()
        oracle.require(row is not None, 'unpublished_candidate')
        manifest, artifact = map(json.loads, row)
        oracle.require(artifact['id'] in job['artifact_ids'] and artifact['manifest_sha256'] == oracle.canonical_hash(manifest), 'stale_candidate')
        self.core.artifact(artifact['id'])
        return copy.deepcopy(manifest['candidates'])

    def save(self, principal, key, job_id, candidate_id, voice_id, *, expected_revision):
        _id(voice_id)
        candidates = self.results(principal, job_id)
        candidate = next((c for c in candidates if c['id'] == candidate_id), None)
        oracle.require(candidate is not None, 'candidate_missing')
        version = copy.deepcopy(candidate['version'])
        version['voice_id'] = voice_id
        # Stable ID binds selection and target; register supplies durable replay/CAS.
        version['id'] = 'saved-' + oracle.canonical_hash([job_id, candidate_id, voice_id])
        return self.core.voices.register(principal, key, version, expected_revision=expected_revision)

    def generate(self, principal, key, context, *, _admission=None, _profiles=None, _dispatch_settings=None):
        context = copy.deepcopy(context)
        plan = context['plan']
        plan.pop('reference_roles', None)
        oracle.require(plan['operation'] == 'speech_generate', 'operation_mismatch')
        version = self.core.voices.version(principal, plan['request']['voice_version_id'])
        plan['voice_versions'] = [version]
        plan['assets'] = [self.core.asset(a) for a in version['reference_asset_ids']]
        from worker.runtime_protocol import execution, dispatch_mode
        spec = execution(context)
        if spec is not None and (spec.engine == 'pocket-tts' or (spec.engine == 'qwen' and spec.task == 'clone')):
            # Core's owned managed asset is the only control-side reference path.
            # Never accept a caller or template's local filesystem override.
            from worker.engines.pocket import _read, _reference_pcm16
            self.core.voices._asset(principal, spec.reference_id)
            raw = _read(self.core.root/'assets'/spec.reference_id, spec.reference_bytes, spec.reference_sha256)
            oracle.require(len(raw) == spec.reference_bytes, 'reference_size')
            _reference_pcm16(raw, spec.reference_rate, spec.reference_frames)
        if spec is not None:
            from worker.engines.native_kokoro import NativeKokoroExecution, JOB_SECONDS
            seconds = JOB_SECONDS if type(spec) is NativeKokoroExecution else (180 if spec.engine=='pocket-tts' else (QWEN_CPU_SECONDS if spec.engine=='qwen' and spec.device=='cpu' else 25))
            _dispatch_settings = (dispatch_mode(spec), min(seconds, plan['resolved']['profile']['limits']['deadline_seconds']))
        return self.core.submit_execution(principal, key, context, _admission=_admission,
                                          _profiles=_profiles, _dispatch_settings=_dispatch_settings)
