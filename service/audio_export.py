"""Trusted operator registration for the finite managed export operation.

Existing installed families have no registration and remain unavailable. This
object is an integration seam, not an enrollment or image qualification shortcut.
"""
import copy
from contracts import validate as oracle
from worker.audio_export import registration, settings, execution


class AudioExports:
    def __init__(self, converter, profile, principals):
        self.converter=registration(converter)
        oracle.validate('profile',profile)
        if not profile['id'].startswith('audio-export-') or profile['compatibility'] or profile['device']!='cpu' or profile['topology']!='networkless_container':
            raise ValueError('export_profile_required')
        self.profile=copy.deepcopy(profile)
        self.principals=frozenset(principals)

    def available(self, principal):
        p=self.profile
        return (principal in self.principals and p['support']=='qualified' and
                p['installation']=='installed' and p['configuration']=='valid' and
                p['readiness']=='ready' and p['verification']=='measured' and
                bool(p['qualification_receipt']))

    def resolve(self, core, principal, request):
        if not self.available(principal): raise ValueError('not_ready')
        request=settings(request)
        from service.domain import owned_asset
        asset=owned_asset(core,principal,request['asset_id'])
        plan=dict(schema_version='1.3.0',operation='audio_export',request=request,project=None,voice_versions=[],assets=[asset],resolved=dict(kind='audio_export',identity=dict(kind='audio_export',converter=copy.deepcopy(self.converter),profile_id=self.profile['id']),profile=oracle.profile_snapshot(self.profile),settings=copy.deepcopy(request),settings_sha256=oracle.canonical_hash(request)))
        context=dict(plan=plan,bundles={},execution=dict(converter=copy.deepcopy(self.converter)))
        execution(context)
        return context
