"""Private first-start native owner. Retained Core/history is unsupported."""
from pathlib import Path

from worker.transport import Refused, canonical
from service import native_package as n


class NativeActivation:
    def __init__(self, record_file, *, configuration_file, templates_file, owner_root, connector=None):
        self.record_file = record_file
        self.inputs = dict(configuration_file=configuration_file, templates_file=templates_file,
                           owner_root=owner_root, connector=connector)
        self.started = False

    def _empty_transport(self, runtime):
        # Even completed imports need the deferred native history policy.
        if (not runtime.admission_open or any(runtime.spool.iterdir()) or
                any(runtime.staging.iterdir())):
            raise Refused('native_retained_work_unsupported')

    def precheck(self, dispatcher):
        if self.started: raise Refused('native_activation_reuse')
        self.started = True
        from service.entrypoint import _INSTALLED_LIFECYCLES
        self.responsibility = dict(owner=dispatcher, closed=False, error=None)
        _INSTALLED_LIFECYCLES[id(self)] = self.responsibility
        root = n.directory(str(dispatcher.root.absolute()))
        if root != Path(self.inputs['owner_root']): raise Refused('native_owner_binding')
        # No DB reader or Linux history policy is used to bless a native history.
        if any(root.iterdir()): raise Refused('native_retained_history_unsupported')
        self.saved = canonical(n.document(self.record_file))
        values = n.native_startup_inputs(self.record_file, **self.inputs)
        runtime = values['runtime']
        if runtime.staging != dispatcher.staging.absolute(): raise Refused('native_staging_binding')
        if (root == runtime.spool or root in runtime.spool.parents or runtime.spool in root.parents or
                root == runtime.staging or root in runtime.staging.parents or runtime.staging in root.parents):
            raise Refused('native_owner_storage_overlap')
        self._empty_transport(runtime)
        if any(root.iterdir()): raise Refused('native_retained_history_unsupported')
        dispatcher.runtime = runtime
        dispatcher.production_contexts = values['production_contexts']
        dispatcher.current_profiles = values['profiles']
        dispatcher.budget = values['budget']
        if dispatcher.persistent_registration is not None: raise Refused('native_linux_registration')
        self.runtime = runtime

    def check_hello(self, hello):
        if (any(hello.get(k) != v for k,v in self.runtime.native.items()) or
                hello.get('quarantined') is not False or
                type(hello.get('launches')) is not int or hello['launches'] != 0):
            raise Refused('native_activation_hello')

    def postcheck(self, dispatcher):
        # Core's lifetime exclusion is held here; all SQLite remains on its owner.
        if not dispatcher.core.fresh_initialized:
            raise Refused('native_retained_history_unsupported')
        if dispatcher.runtime is not self.runtime: raise Refused('native_owner_binding')
        before, _, _ = n._enrollment_snapshot(**{k:v for k,v in self.inputs.items() if k != 'connector'})
        if canonical(before) != self.saved or canonical(n.document(self.record_file)) != self.saved:
            raise Refused('native_enrollment_changed')
        self._empty_transport(self.runtime)
        self.check_hello(self.runtime.call('hello', {}))
        after, _, _ = n._enrollment_snapshot(**{k:v for k,v in self.inputs.items() if k != 'connector'})
        if canonical(after) != self.saved or canonical(n.document(self.record_file)) != self.saved:
            raise Refused('native_enrollment_changed')
        self._empty_transport(self.runtime)

    def finished(self, dispatcher, cleanup_failed, error):
        # Keep a strong owner obligation on timeout/failed close; never retry it.
        from service.entrypoint import _INSTALLED_LIFECYCLES
        if not hasattr(self, 'responsibility') or self.responsibility['owner'] is not dispatcher: return
        self.responsibility['error'] = error
        if not cleanup_failed:
            self.responsibility['closed'] = True
            _INSTALLED_LIFECYCLES.pop(id(self), None)
