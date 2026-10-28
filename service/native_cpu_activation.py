"""Closed CPU activation: owner-held factory, exact source and retained proofs."""
from pathlib import Path
from worker.transport import Refused,canonical
from service.native_activation import NativeActivation,completed_history


class NativeCPUActivation(NativeActivation):
    def __init__(self,prepared):
        from service.native_cpu_install import Prepared
        if type(prepared) is not Prepared:raise Refused('native_cpu_factory_required')
        self.prepared=prepared;self.started=False
        self.session=prepared.qualification

    def run(self,function):
        if self.session is None:return function()
        with self.session.scope():return function()

    def precheck(self,dispatcher):
        if self.started:raise Refused('native_activation_reuse')
        self.started=True
        from service.entrypoint import _INSTALLED_LIFECYCLES
        self.responsibility=dict(owner=dispatcher,closed=False,core_closed=False,error=None,prepared=self.prepared)
        _INSTALLED_LIFECYCLES[id(self)]=self.responsibility
        self.saved=self.prepared.revalidate()
        root=Path(dispatcher.root).absolute()
        if root!=self.prepared.control or Path(dispatcher.staging).absolute()!=self.prepared.staging or dispatcher.persistent_registration is not None:
            raise Refused('native_cpu_owner_binding')
        self.was_empty=not any(root.iterdir())
        if not self.was_empty and not (root/'core.sqlite').is_file():raise Refused('native_retained_history_unsupported')
        dispatcher.runtime=self.runtime=self.prepared.runtime
        dispatcher.production_contexts=self.prepared.contexts
        dispatcher.current_profiles=self.prepared.profiles
        dispatcher.budget=self.prepared.budget
        if self.was_empty:self._empty_transport(self.runtime)

    def retained_check(self,core):
        if self.was_empty!=core.fresh_initialized:raise Refused('native_retained_history_unsupported')
        core._native_retention_runtime=self.runtime
        core._native_retention_templates=dict(version=1,registrations=[])
        if core.fresh_initialized:self.expected_launches=0
        else:
            # The factory owns the actual journal; no boot or receipt learning.
            runtimes={r[0] for r in core.db.execute('SELECT runtime FROM dispatches')}
            records=self.prepared.supervisor.journal.value['records']
            if runtimes!={r['id'] for r in records}:raise Refused('native_cpu_history_coverage')
            self.expected_launches=completed_history(core,self.runtime,_cpu=True)

    def postcheck(self,dispatcher):
        if dispatcher.runtime is not self.runtime or self.saved!=self.prepared.revalidate():raise Refused('native_cpu_installation_changed')
        self.check_hello(self.runtime.call('hello',{}))
        if self.was_empty:
            from service.native_cpu_configuration import VOICE,ALIAS
            principal=self.prepared.config['principal']
            version=self.prepared.contexts.preset_version(principal,'af_heart',VOICE)
            self.prepared.default_version=version['id']
            dispatcher.core.voices.register(principal,'native-setup-preset',version,expected_revision=0)
            voice=dispatcher.core.voices.get(principal,VOICE)
            dispatcher.core.voices.aliases(principal,'native-setup-default',VOICE,[ALIAS],expected_revision=voice['revision'])
        if not self.runtime.admission_open:raise Refused('native_retained_work_unsupported')

    def check_hello(self,hello):
        if (any(hello.get(k)!=v for k,v in self.runtime.native.items()) or hello['quarantined'] or
                hello['launches']!=self.expected_launches or canonical(hello['inventory'])!=canonical(self.prepared.inventory_wire)):
            raise Refused('native_cpu_activation_hello')

    def finished(self,dispatcher,cleanup_failed,error):
        if not hasattr(self,'responsibility') or self.responsibility['owner'] is not dispatcher:return
        self.responsibility['error']=error
        self.responsibility['core_closed']=not cleanup_failed

    def release(self):
        from service.entrypoint import _INSTALLED_LIFECYCLES
        if hasattr(self,'responsibility') and self.responsibility['core_closed']:
            self.responsibility['closed']=True
            _INSTALLED_LIFECYCLES.pop(id(self),None)
