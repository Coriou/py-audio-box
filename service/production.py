"""Trusted synthetic production domain; no HTTP, scheduler or engine readiness.

Contexts are operator-owned resolution snapshots, never public request payloads.
Candidate metadata is read exclusively from Core's committed publication.
"""
import copy
import json
from contracts import validate as oracle
from service.voice_store import _id


class Production:
    def __init__(self, core):
        self.core = core

    def prepare(self, principal, key, context, *, _admission=None, _profiles=None, _dispatch_settings=None):
        context = copy.deepcopy(context)
        plan = context['plan']
        oracle.require(plan['operation'] == 'voice_prepare', 'operation_mismatch')
        asset_id = plan['request']['asset_id']
        self.core.voices._asset(principal, asset_id)
        plan['assets'] = [self.core.asset(asset_id)]
        token = oracle.canonical_hash([principal, key, plan['request']])
        reference = 'reference-' + token
        plan['resolved']['reuse_configuration']['reference_asset_id'] = reference
        audition_text=plan.get('reference_roles',{}).get('audition_text','Synthetic fixture audition.')
        plan['reference_roles'] = dict(version='reference-roles-v1', reference_kind='prepared_reference',
            reference_asset_id=reference, audition_asset_id='audition-'+token,
            audition_text=audition_text)
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
        from worker.runtime_protocol import execution
        spec = execution(context)
        if spec is not None:
            _dispatch_settings = ('kokoro', min(25, plan['resolved']['profile']['limits']['deadline_seconds']))
        return self.core.submit_execution(principal, key, context, _admission=_admission,
                                          _profiles=_profiles, _dispatch_settings=_dispatch_settings)
