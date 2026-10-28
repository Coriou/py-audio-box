"""Owned preset catalogue for the separate native CPU execution family."""
import copy
import json
from contracts import validate as oracle
from lib.spoken_text import freeze
from service.execution_context import KokoroContexts
from worker import native_cpu_protocol as p
from worker.transport import Refused, canonical


class NativeCPUContexts(KokoroContexts):
    def __init__(self,contexts,local_bundles):
        super().__init__(contexts,local_bundles)
        for raw in self._contexts.values():p.native_execution(json.loads(raw))
        self.availability=None

    def bind_availability(self,runtime):
        from service.native_cpu_runtime import NativeCPURuntime
        if type(runtime) is not NativeCPURuntime:raise Refused('native_cpu_runtime_required')
        self.availability=runtime.check_availability

    def _current(self,context):
        if self.availability is None:raise Refused('native_availability_required')
        self.availability(context)

    def _preset_context(self,principal,preset,voice_id):
        if principal not in self._contexts:raise Refused('not_ready')
        context=p.owned_preset_context(json.loads(self._contexts[principal]),principal,preset,voice_id)
        self._current(context)
        return context

    def preset_version(self,principal,preset,voice_id):
        return self._preset_context(principal,preset,voice_id)['plan']['voice_versions'][0]

    def resolve_registered(self,principal,request,version):
        oracle.validate('speech_generate_request',request)
        context=self._preset_context(principal,version['configuration'].get('preset'),version['voice_id'])
        if canonical(context['plan']['voice_versions'])!=canonical([version]):raise Refused('saved_version_required')
        context['plan']['request']=copy.deepcopy(request)
        freeze(context['plan']);p.native_execution(context)
        return context

    def resolve(self,principal,operation,request):
        if operation!='speech_generate' or principal not in self._contexts:raise Refused('not_ready')
        template=json.loads(self._contexts[principal]);version=template['plan']['voice_versions'][0]
        return self.resolve_registered(principal,request,version)

    def capabilities(self,principal):
        if principal not in self._contexts:return None
        context=json.loads(self._contexts[principal])
        session=getattr(self,'qualification',None)
        return dict(engine='kokoro',device='cpu',clone_prepare=False,profile=context['plan']['resolved']['profile'],
            operations=['voice_preset_register','speech_generate','job_get','job_cancel','artifact_get'],
            qualification_session=None if session is None else dict(max_jobs=session.value['max_jobs'],
                remaining_seconds=max(0,session.value['started_monotonic']+session.value['seconds']-session.clock())),
            next_action='Resolve default.narration; speech requires the explicit finite qualification session. Installation is not speech qualification.')
