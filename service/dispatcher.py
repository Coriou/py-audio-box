"""Internal authenticated execution dispatcher with explicit installed activation.

The owner loop alone uses Core/SQLite. A single executor performs bounded UDS I/O;
its detached replies are consumed as commands on that loop, never as DB callbacks.
"""
from concurrent.futures import Future, ThreadPoolExecutor
import copy
import json
from pathlib import Path
import queue
import sqlite3
import threading
import time

from service.core import Core, encode
from service.persistence import PublicationQuarantine, check_operator_resolved
from service.reconciliation import Reconciliation
from service.admission import Admission
from service.dispatch_store import DispatchStore
from service.supervisor_runtime import SupervisorRuntime
from worker.lifecycle import State


class Dispatcher:
    def __init__(self, root, *, ipc='/ipc', staging='/staging', runtime=None, fault=lambda _:None, budget=None, profiles=None, persistent_registration=None, production_contexts=None, installed_activation=None, audio_exports=None):
        self.audio_exports = audio_exports
        self.installed_activation = installed_activation
        self.installed_checked = False
        self.closing = False
        self.production_contexts = production_contexts
        self.persistent_registration = persistent_registration
        self.root, self.staging = Path(root), Path(staging)
        self.runtime = runtime or SupervisorRuntime(ipc)
        self.fault = fault
        self.budget, self.current_profiles = budget, profiles
        self.commands=queue.Queue(64)
        self._cleanup_failed = False
        self.initialized=Future()
        from service.native_cpu_activation import NativeCPUActivation
        target=(lambda:installed_activation.run(self._run)) if type(installed_activation) is NativeCPUActivation else self._run
        self.thread=threading.Thread(target=target,name='control-writer',daemon=True)
        self.thread.start()
        self.initialized.result(timeout=15 if installed_activation is not None else 5)

    def call(self, op, *args, timeout=3, **kwargs):
        if op not in ('submit','job','jobs','events','artifact','asset','cancel','status','replace','inventory_page','reconcile_startup','close','voice_list','voice_get','voice_version','voice_resolve','voice_summaries','voice_version_summaries','asset_ingest','asset_metadata','asset_content','configured_presets','configured_capabilities','configured_profiles','production_preset_register','production_prepare','production_generate','production_audio_export','audio_export_available','audio_export_availability','production_save','production_results','production_job','production_cancel','production_artifact','production_jobs','production_events','production_retry','production_aliases','production_archive','production_profile'):
            raise ValueError('unknown_control_command')
        if op == 'asset_ingest':
            from service.ingest import MAX_BYTES
            if len(args) != 2 or type(args[1]) is not bytes or kwargs:
                raise ValueError('invalid_request')
            if len(args[1]) > MAX_BYTES: raise ValueError('resource_limit')
        future=Future()
        self.commands.put_nowait((op,copy.deepcopy(args),copy.deepcopy(kwargs),future))
        return future.result(timeout=timeout)

    def close(self):
        if self._cleanup_failed: raise RuntimeError('control_cleanup_failed')
        self.call('close');self.thread.join(timeout=4)
        if self.thread.is_alive():raise RuntimeError('control_close_timeout')
        if self._cleanup_failed: raise RuntimeError('control_cleanup_failed')

    def _run(self):
        core=None
        startup_error=None
        try:
            if hasattr(self.runtime,'attach_owner'):self.runtime.attach_owner()
            from service.recovery import startup_read, check_completion_hello, _startup_read_lock
            # Hold the old owner's exclusion through all private readers. Release
            # before Core's independent acquisition (never double-acquire flock).
            # A competing owner in that gap either makes Core refuse, or completes
            # before Core repeats the deployment/completion/hello binding below.
            if self.installed_activation is not None:
                from service.entrypoint import InstalledActivation
                from service.native_activation import NativeActivation
                from service.native_cpu_activation import NativeCPUActivation
                if type(self.installed_activation) not in (InstalledActivation, NativeActivation, NativeCPUActivation):
                    raise ValueError('checked_installed_activation_required')
                self.installed_activation.precheck(self)
                if hasattr(self.runtime, 'attach_owner'): self.runtime.attach_owner()
                startup = None
            else:
                with _startup_read_lock(self.root) as readable:
                    startup = startup_read(self.root) if readable else None
                    hello = None
                    if startup is not None or self.persistent_registration is not None:
                        hello = self.runtime.call('hello', {})
                        if startup is not None and hello.get('maintenance')!=startup['completion']:
                            raise ValueError('b2_cross_volume_completion_mismatch')
                        if self.persistent_registration is None:
                            raise ValueError('persistent_registration_required_before_startup')
                        from service.recovery import enrollment_gate
                        from service.admission import DEFAULT_BUDGET
                        enrollment_gate(self.root, self.persistent_registration,
                            hello, self.current_profiles,
                            DEFAULT_BUDGET if self.budget is None else self.budget)
            from service.native_activation import NativeActivation
            from service.native_cpu_activation import NativeCPUActivation
            installed_family = getattr(self.installed_activation, "expectation", None)
            core=self.core=Core(self.root, _native_activation=(self.installed_activation
                if type(self.installed_activation) in (NativeActivation,NativeCPUActivation) else None),
                _installed_family=installed_family,
                _publication_creation=2 if installed_family is not None else 1 if self.installed_activation is not None or startup is not None else 2)
            if self.installed_activation is not None:
                self.installed_activation.postcheck(self)
                self.installed_checked = True
            elif startup is not None:
                if core.db.execute('SELECT id FROM p1_installation').fetchall()!=[(startup['deployment_id'],)]:
                    raise ValueError('startup_deployment_changed')
                second_hello = self.runtime.call('hello', {})
                check_completion_hello(core.db, second_hello)
                if second_hello!=hello: raise ValueError('startup_hello_changed')
            self.enrolled=bool(core.db.execute("SELECT 1 FROM sqlite_master WHERE name='p1_installation'").fetchone())
            if startup is None and self.enrolled and not self.installed_checked:
                raise ValueError('startup_deployment_changed')
            if self.installed_checked:
                from service.native_activation import NativeActivation
                from service.native_cpu_activation import NativeCPUActivation
                self.enrolled = self.enrolled or type(self.installed_activation) in (NativeActivation,NativeCPUActivation)
            bind = getattr(self.production_contexts, 'bind_availability', None)
            if bind is not None: bind(self.runtime)
            self.admission=Admission(core,self.staging,self.budget,private_spool=getattr(self.runtime,'spool',None),
                native_availability=getattr(self.runtime, 'check_availability', None))
            self.store=DispatchStore(core,self.admission)
            if hasattr(self.runtime,'recover_imports'):self.runtime.recover_imports(core)
            self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='bounded-lifecycle')
            self.pending=None;self.reconciled=False;self.quarantined=False
            # A former local/foundation attempt has no supervisor death proof.
            self.orphaned=bool(self.core.db.execute("SELECT 1 FROM jobs WHERE json_extract(job,'$.state') IN ('running','cancel_requested') AND id NOT IN (SELECT job_id FROM dispatches) LIMIT 1").fetchone())
            monitor_state=self.admission.store.state()
            self.persistence_fault=monitor_state not in ('healthy','fault')
            self.resource_fault=monitor_state!='healthy' or bool(core.db.execute('SELECT 1 FROM resource_faults').fetchone())
            self.stop_obligation=self.resource_fault
            self.resource_stop_sent=set()
            self.admission.store.on_fault=self._sync_resource_fault
            self.error=None;self.next_poll=0;self.stopping=False
            self.profiles=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['profiles']
            for p in self.profiles:
                p.update(support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='E1-W2d-fixed-synthetic-only')
            if self.current_profiles is not None:self.profiles=copy.deepcopy(self.current_profiles)
            self.orphaned = self.orphaned or self.admission.unaccounted
            self.inventory = None
            self.integrity = dict(complete=False, blocked=True, reasons=['inventory_unavailable'], counts={})
            try:
                self.inventory = Reconciliation(core, self.staging)
                self.integrity = self.inventory.reconcile_startup()
            except (OSError, sqlite3.Error, ValueError):
                pass  # Diagnostic failure is fail-closed, but the owner must service stops.
            self.integrity_fault = self.integrity['blocked']
            if not self.enrolled:
                self._release_startup_cancelled()
            self.initialized.set_result(True)
            while not self.stopping:
                try:
                    item=self.commands.get(timeout=.02)
                except queue.Empty:item=None
                if item:
                    op,args,kwargs,future=item
                    try:future.set_result(copy.deepcopy(self._command(op,*args,**kwargs)))
                    except Exception as exc:
                        self._sync_resource_fault()
                        future.set_exception(exc)
                if self.stopping:break
                try:self._tick()
                except Exception as exc:
                    # Persistence/verification faults retain durable ownership and evidence.
                    self._sync_resource_fault()
                    self._native_cpu_trace('tick_refused',error=exc)
                    self.error=type(exc).__name__;self.quarantined=True
                    self.next_poll=time.monotonic()+.2
            if self.pending:self.pending[0].result(timeout=3)
        except BaseException as exc:
            if not self.initialized.done():startup_error=exc
            else:self.error=type(exc).__name__
        finally:
            # Failed initialization is observable only after both owned cleanups.
            # The constructor retains its existing five-second wait bound.
            try:
                if hasattr(self,'pool'):self.pool.shutdown(wait=True,cancel_futures=True)
            except BaseException as cleanup:
                self._cleanup_failed = True
                if startup_error is not None: startup_error.add_note('startup pool cleanup failed: '+type(cleanup).__name__)
            finally:
                try:
                    if core:core.close()
                except BaseException as cleanup:
                    self._cleanup_failed = True
                    if startup_error is not None: startup_error.add_note('startup Core cleanup failed: '+type(cleanup).__name__)
                finally:
                    from service.native_activation import NativeActivation
                    from service.native_cpu_activation import NativeCPUActivation
                    if type(self.installed_activation) in (NativeActivation,NativeCPUActivation):
                        self.installed_activation.finished(self, self._cleanup_failed, startup_error)
                    if startup_error is not None:self.initialized.set_exception(startup_error)

    def _activation_open(self):
        # One closed predicate for commands, scheduler and queued release.
        return not self.enrolled or (self.installed_checked and self.reconciled and
            not self.quarantined and not self.orphaned and not self.resource_fault and
            not self.persistence_fault and not self.integrity_fault)

    def _release_startup_cancelled(self):
        with self.core.transaction():
            for reservation in self.admission.store.rows():
                if (self._activation_open() and not self.orphaned and not self.resource_fault
                        and not self.integrity_fault and not self.persistence_fault
                        and reservation['state']=='queued'
                        and self.core.job(reservation['id'])['state']=='cancelled'):
                    self.admission.store.release_tx(reservation['id'],unstarted=True)

    def _lifecycle_ready(self):
        return (self._activation_open() and self.reconciled and not self.closing and
                getattr(self.runtime,'admission_open',True) and
                not self.orphaned and not self.quarantined and not self.resource_fault and
                not self.persistence_fault and not self.integrity_fault)

    def _native_cpu_active(self):
        from service.native_cpu_activation import NativeCPUActivation
        return self.installed_checked and type(self.installed_activation) is NativeCPUActivation and self._lifecycle_ready()

    def _local_qwen_active(self):
        """Execution authority for the exact installed, unqualified Qwen lane."""
        from service.entrypoint import InstalledActivation
        from service.execution_context import QwenContexts
        from service.installed_qwen_daemon import FAMILY
        from service.qwen_local_lane import context_allowed, profile_allowed
        from contracts import validate as oracle
        activation = self.installed_activation
        contexts = self.production_contexts
        if (not self.installed_checked or type(activation) is not InstalledActivation or
                activation.family is not FAMILY or getattr(self.runtime, 'family', None) is not FAMILY or
                type(contexts) is not QwenContexts or contexts.installed_family is not FAMILY or
                not contexts.service or len(self.profiles) != 1 or
                activation.checked is None or activation.prepared != (self.runtime, self.root.absolute()) or
                self.profiles != activation.configuration['profiles'] or
                not self._lifecycle_ready()):
            return False
        profile = self.profiles[0]
        try:
            configured = {(principal, name): context for principal, group in activation.configuration['contexts'].items()
                          for name, context in group.items()}
            return (profile_allowed(profile) and bool(configured) and
                    set(contexts._contexts) == set(configured) and
                    all(oracle.profile_snapshot(profile) == context['plan']['resolved']['profile'] and
                        context == configured[key] and context_allowed(context, profile)
                        for key, context in ((key, json.loads(raw)) for key, raw in contexts._contexts.items())))
        except (KeyError, TypeError, ValueError):
            return False

    def _execution_active(self):
        return self._ready() or self._native_cpu_active() or self._local_qwen_active()

    def _ready(self):
        return self._lifecycle_ready() and (not self.installed_checked or all(
                    p['support']=='qualified' and p['installation']=='installed' and
                    p['readiness']=='ready' and p['verification']=='measured'
                    for p in self.profiles))

    def _production_command(self, op, principal, payload, *, key=None):
        from contracts import validate as oracle
        from service.production import Production
        from service.catalogue import OPERATIONS
        from service.voice_store import _id
        operation = {'production_preset_register':'voice_preset_register', 'production_prepare':'voice_prepare', 'production_generate':'speech_generate', 'production_audio_export':'audio_export',
                     'production_save':'voice_save', 'production_results':'voice_results',
                     'production_job':'job_get', 'production_cancel':'job_cancel',
                     'production_artifact':'artifact_get', 'production_jobs':'jobs_list', 'production_events':'job_events',
                     'production_retry':'job_retry', 'production_aliases':'voice_aliases', 'production_archive':'voice_archive', 'production_profile':'profile_get'}[op]
        oracle.validate(OPERATIONS[operation]['request'], payload)
        if OPERATIONS[operation]['idempotency']: _id(key)
        domain = Production(self.core)
        def owned(job_id):
            job = self.core.job(job_id)
            if job['principal_id'] != principal: raise ValueError('not_found')
            return job
        if op == 'production_profile':
            from service.production_selection import discovery
            for profile in discovery(self):
                if profile['id'] == payload['profile_id']: return profile
            raise ValueError('not_found')
        if op in ('production_aliases', 'production_archive'):
            method = self.core.voices.aliases if op == 'production_aliases' else self.core.voices.archive
            return method(principal, key, **payload)
        if op == 'production_retry':
            old = owned(payload['job_id'])
            oracle.require(payload['input_sha256'] == old['input_sha256'], 'idempotency_conflict')
            _, context = self.core._load(old['id'])
            from worker.native_family import frozen_native
            if hasattr(self.runtime, 'native') or frozen_native(context):
                raise ValueError('unsupported_control')
            # Replays precede authority refresh, but remain bound to the source
            # job even when two sources have byte-identical frozen inputs.
            job = self.core.execution_duplicate(principal, old['operation'], key,
                context['plan']['request'], context=context, retry_of=old['id'])
            if job is None:
                oracle.require(old['state'] in ('failed','cancelled') and all(a['reaped'] for a in old['attempts']), 'unsupported_control')
                if not (self._ready() or self._local_qwen_active()): raise ValueError('not_ready')
                if self.core.db.execute("SELECT 1 FROM dispatches WHERE job_id=? AND (acked=0 OR phase='unknown')", (old['id'],)).fetchone():
                    raise ValueError('not_ready')
                if self.core.db.execute('SELECT 1 FROM publications WHERE job_id=?',(old['id'],)).fetchone(): raise ValueError('unsupported_control')
                from service.persistence import check_operator_resolved
                check_operator_resolved(self.core.db,old['id'])
                from service.production_selection import ContextComposition, resolve_owned, choose
                request = context['plan']['request']
                if old['operation']=='audio_export':
                    if (self.installed_checked and not self._checked_export_family()) or self.audio_exports is None: raise ValueError('not_ready')
                    self.core.require_export_format('audio_export')
                    current=self.audio_exports.resolve(self.core,principal,request)
                elif self.production_contexts is None: raise ValueError('not_ready')
                elif isinstance(self.production_contexts, ContextComposition):
                    current = self.production_contexts.resolve_selected(self,principal,old['operation'],request)
                else:
                    current = resolve_owned(self.production_contexts,self.core,principal,old['operation'],request)
                    current = choose(self,[('configured',current)],request)
                # Resolve current authority, never replace the historical plan.
                oracle.require(current.get('execution') == context.get('execution'), 'unsupported_control')
                for field in ('identity','profile','settings','settings_sha256'):
                    oracle.require(current['plan']['resolved'][field] == context['plan']['resolved'][field], 'unsupported_control')
                settings = self.core.db.execute('SELECT fixture,seconds FROM dispatch_settings WHERE job_id=?',(old['id'],)).fetchone()
                if settings is None: raise ValueError('unsupported_control')
                job = self.core.submit_execution(principal,key,context,retry_of=old['id'],
                    _admission=self.admission,_profiles=self.profiles,_dispatch_settings=tuple(settings))
            return dict(job_id=job['id'],input_sha256=job['input_sha256'],state='queued',status_url='/v1/jobs/'+job['id'])
        if op in ('production_jobs', 'production_events'):
            from service.job_control import cursor_decode, cursor_encode
            job_id = payload.get('job_id', '')
            if job_id: job = owned(job_id)
            kind = 'events' if job_id else 'jobs'
            after = cursor_decode(payload['cursor'], principal, kind, job_id)
            if after:
                query = ('SELECT 1 FROM events WHERE seq=? AND job_id=?' if job_id else
                         'SELECT 1 FROM jobs WHERE seq=? AND principal=?')
                if not self.core.db.execute(query, (after, job_id or principal)).fetchone():
                    raise ValueError('invalid_request')
            rows = self.core.events(job_id, after, payload['limit']) if job_id else self.core.jobs(principal, after, payload['limit'])
            token = lambda seq: cursor_encode(seq, principal, kind, job_id)
            if job_id:
                return dict(job=job, events=[dict(cursor=token(r['cursor']),state=r['state']) for r in rows],
                            next_cursor=token(rows[-1]['cursor']) if rows else payload['cursor'])
            return dict(items=[r['job'] for r in rows],next_cursor=token(rows[-1]['cursor']) if len(rows)==payload['limit'] else None)
        if op == 'production_job': return owned(payload['job_id'])
        if op == 'production_artifact':
            artifact = self.core.artifact(payload['artifact_id'])
            owned(artifact['job_id'])
            return artifact
        if op == 'production_results':
            owned(payload['job_id'])
            return dict(schema_version='1.3.0', job_id=payload['job_id'], candidates=domain.results(principal, payload['job_id']))
        if op == 'production_cancel':
            owned(payload['job_id'])
            # Durable intent binds the key before invoking existing cancel logic.
            # Incomplete intent after restart can only resume that exact cancellation.
            with self.core.transaction():
                fingerprint = oracle.fingerprint(principal, 'job_cancel', payload)
                row = self.core.db.execute("SELECT fingerprint FROM catalogue_requests WHERE principal=? AND operation='job_cancel' AND key=?", (principal,key)).fetchone()
                if row and row[0] != fingerprint: raise ValueError('idempotency_conflict')
                if not row: self.core.db.execute('INSERT INTO catalogue_requests VALUES(?,?,?,?,?)', (principal,'job_cancel',key,fingerprint,encode(payload)))
            return self._command('cancel', payload['job_id'])
        if op == 'production_preset_register':
            if not self.installed_checked or not self._execution_active() or self.production_contexts is None:
                raise ValueError('not_ready')
            resolver = getattr(self.production_contexts, 'preset_version', None)
            if resolver is None: raise ValueError('not_ready')
            version = resolver(principal, payload['preset'], payload['voice_id'])
            return domain.register_preset(principal, key, version, expected_revision=payload['expected_revision'])
        if op == 'production_save':
            if self.installed_checked and not (self._ready() or self._local_qwen_active()): raise ValueError('not_ready')
            if not self._activation_open() or self.closing or self.integrity_fault or self.persistence_fault: raise ValueError('not_ready')
            owned(payload['job_id'])
            return domain.save(principal, key, payload['job_id'], payload['candidate_id'], payload['voice_id'], expected_revision=payload['expected_revision'])
        if op == 'production_audio_export':
            job=self.core.execution_duplicate(principal,operation,key,payload)
            if job is None:
                if (not self._activation_open() or self.closing or not self.reconciled or self.quarantined or self.resource_fault or self.integrity_fault or self.persistence_fault or not getattr(self.runtime,'admission_open',True) or self.audio_exports is None or (self.installed_checked and (not self._checked_export_family() or not self._ready()))):
                    raise ValueError('not_ready')
                self.core.require_export_format('audio_export')
                context=self.audio_exports.resolve(self.core,principal,payload)
                job=domain.audio_export(principal,key,context,_admission=self.admission,_profiles=self.profiles)
            return dict(job_id=job['id'],input_sha256=job['input_sha256'],state='queued',status_url='/v1/jobs/'+job['id'])
        # Duplicate lookup retains the original frozen identity and reservation.
        job = self.core.execution_duplicate(principal, operation, key, payload)
        if job is None:
            if not self._activation_open() or self.closing or not self.reconciled or self.quarantined or self.resource_fault or self.integrity_fault or not getattr(self.runtime,'admission_open',True):
                raise ValueError('not_ready')
            if self.production_contexts is None: raise ValueError('not_ready')
            from service.production_selection import ContextComposition, resolve_owned, choose
            if isinstance(self.production_contexts, ContextComposition):
                context = self.production_contexts.resolve_selected(self, principal, operation, payload)
            else:
                context = resolve_owned(self.production_contexts, self.core, principal, operation, payload)
                context = choose(self, [('configured', context)], payload)
            method = domain.prepare if op == 'production_prepare' else domain.generate
            job = method(principal, key, context, _admission=self.admission,
                         _profiles=self.profiles, _dispatch_settings=('silence',8))
        return dict(job_id=job['id'], input_sha256=job['input_sha256'], state='queued', status_url='/v1/jobs/'+job['id'])

    def _native_cpu_relation(self):
        from service.native_cpu_activation import NativeCPUActivation
        from service.native_cpu_runtime import NativeCPURuntime
        activation=type(getattr(self,'installed_activation',None)) is NativeCPUActivation
        runtime=type(getattr(self,'runtime',None)) is NativeCPURuntime
        if activation!=runtime:raise ValueError('native_cpu_runtime_binding')
        return activation

    def _checked_export_family(self):
        if self._native_cpu_relation():return False
        from service.installed_export import FAMILY
        return (self.installed_checked and getattr(self.installed_activation, "family", None) is FAMILY
                and self.core.publication_format == 2 and self.core._installed_family is not None)

    def _command(self,op,*args,**kwargs):
        if op=='audio_export_availability':
            if (self.installed_checked and (not self._checked_export_family() or not self._ready())) or not self.audio_exports or not self.audio_exports.available(args[0]): return 'unavailable_converter'
            return 'implemented' if self.core.publication_format==2 else 'publication_format_migration_required'
        if op=='audio_export_available': return self._command('audio_export_availability',*args)=='implemented'
        if op.startswith('production_'): return self._production_command(op,*args,**kwargs)
        if op=='asset_ingest':
            if self.installed_checked and not (self._ready() or self._local_qwen_active()): raise ValueError('not_ready')
            from service.ingest import ingest_local
            if len(args)!=2 or type(args[1]) is not bytes or kwargs: raise ValueError('invalid_request')
            return ingest_local(self.core,*args)
        if op in ('asset_metadata','asset_content'):
            from service.domain import owned_asset
            return owned_asset(self.core,*args,content=op=='asset_content',**kwargs)
        if op=='configured_profiles':
            from service.production_selection import discovery
            return discovery(self)
        if op=='configured_capabilities':
            describe = getattr(self.production_contexts, 'capabilities', None)
            configured = describe(args[0]) if describe else None
            if configured:
                configured['profile'] = next((copy.deepcopy(p) for p in self.profiles if p['id'] == configured['profile']['id']), None)
                if configured['profile'] is None: return None
            return configured
        if op=='configured_presets':
            describe = getattr(self.production_contexts, 'presets', None)
            return describe(args[0]) if describe else []
        if op=='voice_summaries':return self.core.voices.summaries(*args,**kwargs)
        if op=='voice_version_summaries':return self.core.voices.version_summaries(*args,**kwargs)
        if op=='voice_list':return self.core.voices.list(*args,**kwargs)
        if op=='voice_get':return self.core.voices.get(*args,**kwargs)
        if op=='voice_version':return self.core.voices.version(*args,**kwargs)
        if op=='voice_resolve':return self.core.voices.resolve(*args,**kwargs)
        if op=='close':
            if not self.installed_checked:
                self.stopping=True
                return None
            self.closing=True
            self.stop_obligation=True
            if not self.pending and not self.store.current(): self.stopping=True
            return None
        if op=='reconcile_startup':
            if self.pending or self.store.current():raise ValueError('inventory_lane_busy')
            if self.inventory is None:raise ValueError('inventory_unavailable')
            self.integrity=self.inventory.reconcile_startup()
            self.integrity_fault = self.integrity_fault or self.integrity['blocked']
            return self.integrity
        if op=='inventory_page':
            if self.inventory is None:raise ValueError('inventory_unavailable')
            try:return self.inventory.inventory_page(*args,**kwargs)
            finally:
                self.integrity=self.inventory.summary
                self.integrity_fault=self.integrity_fault or self.integrity['blocked']
                if self.integrity_fault:self.stop_obligation=True
        if op=='status':
            try:row=self.store.current()
            except ValueError:row=None;self.error='dispatch_identity_unreadable'
            return dict(retry_available=not hasattr(self.runtime,'native'),reconciled=self.reconciled,quarantined=self.quarantined,
                phase=row['phase'] if row else 'idle',runtime_id=row['runtime'] if row else None,
                resource_activation=self._ready() or self._local_qwen_active(),
                persistent_activation=self.enrolled and self._execution_active(),error=self.error,
                resource_fault=self.resource_fault,persistence_fault=self.persistence_fault,
                stop_obligation=self.stop_obligation,
                reservations=self.admission.store.rows(), integrity=self.integrity,
                integrity_fault=self.integrity_fault)
        if op=='cancel':
            if not self._activation_open(): raise ValueError('persistent_activation_closed_until_C1')
            if self.installed_checked and self.core.job(args[0])['state']=='queued' and not self._execution_active(): raise ValueError('not_ready')
            if self.integrity_fault and self.core.job(args[0])['state']=='queued':raise ValueError('integrity_blocks_release')
            job=self.core.request_cancel(*args,**kwargs,_cancel_unstarted=True,_admission=self.admission)
            self._interrupt_transfer(job['id'])
            self.next_poll=0
            return job
        if op in ('submit','replace'):
            # Idempotent lookup is a read, including during quarantine/reconnect.
            if op=='submit':
                if len(args)!=2 or not set(kwargs)<= {'retry_of','fixture','seconds'}:raise ValueError('closed_synthetic_command')
                context=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['context']
                existing=self.core.execution_duplicate(args[0], 'speech_generate', args[1],
                    context['plan']['request'], context=context, retry_of=kwargs.get('retry_of'))
                if existing is not None:return existing
            if not self._activation_open(): raise ValueError('persistent_activation_closed_until_C1')
            if self.installed_checked: raise ValueError('installed_production_commands_required')
            if self.closing or not self.reconciled or self.quarantined or self.resource_fault or self.integrity_fault or not getattr(self.runtime,'admission_open',True):raise ValueError('startup_or_quarantine')
            if op=='replace':
                if self.pending or self.store.current():raise ValueError('lane_busy')
                job_id=args[0]
                check_operator_resolved(self.core.db,job_id)
                row=self.store.allocate(self.profiles,'silence',8,replacement=job_id)
                self.fault('prespawn')
                self._prepare(row)
                return self.core.job(row['job_id'])
            # Fixed installed internal fixture; caller supplies no context/profile/plan.
            principal,key=args
            if not set(kwargs)<= {'retry_of','fixture','seconds'}:raise ValueError('closed_synthetic_command')
            fixture=kwargs.get('fixture','silence');seconds=kwargs.get('seconds',8)
            if fixture not in ('silence','busy','descendants','permissions') or type(seconds)!=int or not 2<=seconds<=25:raise ValueError('fixed_fixture_required')
            context=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['context']
            job=self.core.submit_synthetic(principal,key,context,retry_of=kwargs.get('retry_of'),
                _dispatch_settings=(fixture,seconds),_admission=self.admission,_profiles=self.profiles)
            return job
        return getattr(self.core,op)(*args,**kwargs)

    def _prepare(self, row):
        if hasattr(self.runtime,'detach'):
            detached=self.runtime.detach(row['authority'],self.core)
            self._launch(row,'prepare',lambda:self.runtime.prepare(row['authority'],row['fixture'],detached))
        else:
            self._launch(row,'prepare',lambda:self.runtime.prepare(row['authority'],row['fixture']))

    def _interrupt_transfer(self, job_id):
        row=self.store.current()
        if row and row['job_id']==job_id and row['runtime'] and hasattr(self.runtime,'interrupt'):
            self.runtime.bind(row['authority'],row['runtime'])
            self.runtime.interrupt(row['runtime'])

    def _qwen_state_poll(self):
        if self._native_cpu_relation():return False
        from service.installed_qwen_daemon import FAMILY
        return (self.installed_checked and
                getattr(self.installed_activation, 'family', None) is FAMILY and
                getattr(self.runtime, 'family', None) is FAMILY)

    def _observation_interval(self,op,value):
        if self._native_cpu_relation():return .03
        from service import installed_pocket_daemon as pd
        return .2 if (getattr(self.runtime,'family',None) is pd.FAMILY and
                       op=='observe' and value.state==State.ALIVE) else .03

    def _native_cpu_trace(self,stage,row=None,op=None,error=None):
        """At most32 distinct source-only events; no request/config/media values."""
        from service.native_cpu_activation import NativeCPUActivation
        if type(getattr(self,'installed_activation',None)) is not NativeCPUActivation:return
        import re,sys
        from service.entrypoint import _startup_diagnostic
        event=dict(stage=stage,op=op,phase=None if row is None else row['phase'])
        if row is not None:
            for key,pattern in (('job_id',r'job-[0-9a-f]{32}'),('runtime',r'[0-9a-f]{32}/[0-9a-f]{32}/[0-9a-f]{32}')):
                value=row.get(key);event[key]=value if isinstance(value,str) and re.fullmatch(pattern,value) else None
        if error is not None:
            self.error=type(error).__name__
            event['error']=_startup_diagnostic(error)
            allowed={'native_shape','native_output','native_control_storage','native_control_uncertain','native_deadline',
                'native_cancelled','native_busy','native_unknown','native_refused','native_owner_required'}
            if type(error).__name__=='Refused' and str(error) in allowed:event['code']=str(error)
        seen=getattr(self,'_cpu_trace_seen',set());self._cpu_trace_seen=seen
        key=json.dumps(event,sort_keys=True)
        if key in seen or len(seen)>=32:return
        seen.add(key)
        print('native_cpu_dispatch:'+key,file=sys.stderr)

    def _launch(self,row,op,fn):
        command=None if op=='poll' else self.store.command(row,'observe' if op=='confirm' else op)
        self.fault('before_'+op)
        self._native_cpu_trace('issued',row,op)
        self.pending=(self.pool.submit(fn),row,op,command)

    def _sync_resource_fault(self):
        store=self.admission.store
        if store.failed or store.state()!='healthy':
            if not self.resource_fault:self.stop_obligation=True
            self.resource_fault=self.quarantined=True
            self.persistence_fault=store.state()!='fault'

    def _resource_check(self, *, publication=False):
        if self.resource_fault:return
        try:
            self.admission.store.observe(lambda:self.admission.check(publication=publication))
        except Exception as exc:
            self.error=str(exc) if isinstance(exc,(OSError,ValueError)) else type(exc).__name__
            self.resource_fault=self.quarantined=self.stop_obligation=True
            self.persistence_fault=self.admission.store.state()!='fault'

    def _service_stop(self,row):
        # One bounded stop request per bound runtime, including after pending I/O.
        # Failed/unknown stop is not death: observation/receipt must still follow.
        if ((self.stop_obligation or (row and self.core.job(row['job_id'])['state']=='cancel_requested'))
                and row and row['runtime'] and not row['outcome']
                and not row['receipt'] and row['runtime'] not in self.resource_stop_sent and not self.pending):
            self._launch(row,'stop',lambda:self.runtime.terminate(row['runtime'],1.5))
            self.resource_stop_sent.add(row['runtime'])
            return True
        return False

    def _tick(self):
        if self.enrolled and not self.installed_checked: return  # Legacy inspect-only.
        if getattr(self.core,'_publication_storage_failed',False):
            self.persistence_fault=self.quarantined=True
            return  # Exact reap preceded publication; storage needs reopen, no live retry.
        self._resource_check()
        if self.pending:
            future,row,op,command=self.pending
            if not future.done():return
            self.pending=None
            try:
                value=future.result()
                if op=='poll' and (value.runtime_id!=row['runtime'] or value.state not in (State.ALIVE,State.STOPPED,State.UNKNOWN)):
                    raise ValueError('unbound_observation')
                if op=='collect' and hasattr(self.runtime,'finalize_collect'):
                    value=self.runtime.finalize_collect(value)
            except Exception as exc:
                self._native_cpu_trace('refused',row,op,exc)
                if op=='poll':command=self.store.command(row,'observe')
                self.store.complete(command,False)
                if op=='ack':self.next_poll=time.monotonic()+.2;return
                self.store.update(row,phase='unknown');self.quarantined=True
                self.reconciled=True;self.next_poll=time.monotonic()+.2;return
            self._native_cpu_trace('completed',row,op)
            self.fault('after_'+op)
            if command is not None:self.store.complete(command,True)
            if op=='poll':
                if (value.state!=State.ALIVE or (hasattr(self.runtime,'import_uncertain') and
                        self.runtime.import_uncertain(row['runtime']))):
                    # A read is only a hint; collect requires durable STOPPED confirmation.
                    self._launch(row,'confirm' if value.state==State.STOPPED else 'observe',
                                 lambda:self.runtime.observe(row['runtime'],1.5));return
            if op=='prepare':
                self.fault('preregistration')
                self.store.update(row,runtime=value,phase='bound')
                self.fault('prestart')
                bound=self.store.current()
                if self.core.job(bound['job_id'])['state']=='cancel_requested':
                    self._interrupt_transfer(bound['job_id'])
                if self._service_stop(bound):return
                self._launch(bound,'start',lambda:self.runtime.start(value));return
            if op=='start':self.store.update(row,phase='observing');self.fault('postspawn')
            elif op in ('observe','confirm'):
                if value.runtime_id!=row['runtime']:raise ValueError('unbound_observation')
                if op=='confirm' and value.state!=State.STOPPED:
                    self.store.update(row,phase='unknown');self.quarantined=True
                    self.reconciled=True;self.next_poll=time.monotonic()+.2;return
                self.reconciled=True
                self.quarantined=value.state==State.UNKNOWN or self.resource_fault or (hasattr(self.runtime,'import_uncertain') and self.runtime.import_uncertain(row['runtime']))
                if value.state==State.STOPPED:
                    if self.persistence_fault and row['receipt']:
                        if self.core.job(row['job_id'])['state'] not in ('succeeded','failed','cancelled'):
                            self._terminal(row,row['receipt'],None)
                    else:
                        self.fault('observation_reap')
                        self._launch(row,'collect',lambda:self.runtime.collect(row['authority'],row['runtime']));return
                self.store.update(row,phase='unknown' if self.quarantined else 'observing')
            elif op=='collect':
                if value is None:raise ValueError('terminal_receipt_missing')
                self._terminal(row,*value)
            elif op=='ack':
                if value!={'acked':True}:raise ValueError('ack_required')
                self.store.update(row,acked=1,phase='acknowledged');self.quarantined=getattr(self,'resource_fault',False)
            self.next_poll=time.monotonic()+self._observation_interval(op,value)
        if self.pending:return
        if self._service_stop(self.store.current()):return
        if time.monotonic()<self.next_poll:return
        row=self.store.current()
        if row:
            if hasattr(self.runtime,'bind') and row['runtime']: self.runtime.bind(row['authority'],row['runtime'])
            if row['outcome']:
                if self.integrity_fault:return
                if self.persistence_fault:return
                self._launch(row,'ack',lambda:self.runtime.call('ack',dict(runtime_id=row['runtime'])));return
            if row['phase']=='intent':
                # Only a new live allocation reaches prepare. Startup never replays intent.
                self.store.update(row,phase='unknown');self.quarantined=True;self.reconciled=True
            if row['runtime'] is None:
                self.quarantined=True;self.reconciled=True;return
            op='poll' if self._qwen_state_poll() and row['phase']=='observing' and not self.quarantined else 'observe'
            self._launch(row,op,lambda:self.runtime.observe(row['runtime'],1.5));return
        if self.closing:
            self.stopping=True
            return
        if not self.reconciled:
            # Validate actual installed supervisor before admitting anything.
            if not hasattr(self,'hello'):
                self.hello=self.pool.submit(self.runtime.call,'hello',{})
            if not self.hello.done():return
            try:
                value=self.hello.result()
                if self.installed_checked: self.installed_activation.check_hello(value)
                self.quarantined=self.orphaned or self.resource_fault or bool(value['quarantined'])
            except Exception:self.quarantined=True
            self.reconciled=True
            if self.installed_checked: self._release_startup_cancelled()
            return
        if self.quarantined or self.integrity_fault or not getattr(self.runtime,'admission_open',True):return
        queued=next((json.loads(r[0]) for r in self.core.db.execute('SELECT job FROM jobs ORDER BY seq') if json.loads(r[0])['state']=='queued'),None)
        if queued:
            settings=self.core.db.execute('SELECT fixture,seconds FROM dispatch_settings WHERE job_id=?',(queued['id'],)).fetchone()
            if settings is None:raise ValueError('fixed_dispatch_settings_required')
            row=self.store.allocate(self.profiles,*settings)
            self.fault('prespawn')
            self._prepare(row)

    def _terminal(self,row,receipt,observation):
        if getattr(self.core,'_publication_storage_failed',False):
            self.persistence_fault=self.quarantined=True
            return  # Only reopen may resolve publication storage ambiguity.
        if hasattr(self.runtime,'import_uncertain') and self.runtime.import_uncertain(row['runtime']):
            self.store.update(row,phase='unknown');self.quarantined=True
            return
        authority=row['authority'];job=self.core.job(row['job_id'])
        from worker.native_family import terminal_receipt, frozen_native
        native=frozen_native(authority.context)
        import_uid=terminal_receipt(receipt,authority=authority,runtime_id=row['runtime'],
            local_import_uid=getattr(self.runtime,'local_import_uid',None))
        if native and import_uid is None: raise ValueError('trusted_native_import_uid_required')
        if self.integrity_fault:
            # Retain a conflicting original receipt; a fresh exact receipt may
            # only fill an empty slot. Neither form authorizes release/ACK here.
            if not row['receipt']:self.store.update(row,receipt=encode(receipt),phase='receipt')
            self.stop_obligation=False
            return
        self.store.update(row,receipt=encode(receipt),phase='receipt')
        self.fault('preverification')
        if job['state'] not in ('succeeded','failed','cancelled'):
            intent=self.core.db.execute('SELECT 1 FROM publications WHERE job_id=?',(job['id'],)).fetchone()
            try:
                # Failed/late attempts may retain incomplete private media. Always
                # account its bytes, but decode only a proposal eligible to publish.
                self._resource_check(publication=bool(intent or (receipt['timely'] is True and observation is not None)))
                if getattr(self,'resource_fault',False):raise ValueError('resource_violation')
                if intent:
                    if receipt['timely'] is not True:raise ValueError('late_publication_intent')
                    stored=json.loads(self.core.db.execute('SELECT artifact FROM publications WHERE job_id=?',(job['id'],)).fetchone()[0])
                    for key in ('generation','attempt_id','owner_id','staging_id'):
                        expected=authority.job['generation'] if key=='generation' else authority.job['attempts'][-1]['id' if key=='attempt_id' else key]
                        if stored[key]!=expected:raise ValueError('stale_publication_intent')
                    self.core._commit_publication(job['id'])
                elif receipt['timely'] is True and observation is not None:
                    self._native_cpu_trace('publication',row,'publish')
                    if native:
                        from datetime import datetime, timezone
                        if datetime.now(timezone.utc)>=datetime.fromisoformat(authority.job['deadline']): raise ValueError('native_import_deadline')
                    self.core._publish(job['id'],dict(message=observation.message,manifest=observation.manifest),
                        fault=self.fault,stage_root=self.staging,engine_uid=import_uid)
                else:self._fail(job,'deadline_or_reaped')
            except (OSError,sqlite3.Error,PublicationQuarantine):
                # Storage failure is not invalid media. Retain intent, ownership,
                # reservation and receipt; a reopened session reconciles DB state.
                self.persistence_fault=True
                self.quarantined=True
                raise
            except (ValueError,AssertionError) as exc:
                self._fail(self.core.job(job['id']),'verification_failed')
        # Crash after success before outcome journal is recoverable from Core terminal.
        job=self.core.job(job['id'])
        if job['state'] not in ('succeeded','failed','cancelled'):raise ValueError('terminal_required')
        self._native_cpu_trace('terminal',row,job['state'])
        self.stop_obligation=False  # Only the exact validated receipt discharges stop.
        if self.persistence_fault:
            # Fault authority could not be journaled: retain the reservation even
            # after receipt. No automatic repair/ACK/release is invented here.
            self.store.update(row,phase='unknown')
            self.quarantined=True
            return
        try:
            with self.core.transaction():
                self.admission.store.release_tx(job['id'],confirmed_reap=True)
                self.core.db.execute("UPDATE dispatches SET outcome=?,phase='ack_pending' WHERE seq=?",(encode(job),row['seq']))
        except (OSError,sqlite3.Error):
            self.core._publication_storage_failed=True
            self.persistence_fault=self.quarantined=True
            raise
        self.fault('durable_outcome')
        self.quarantined=getattr(self,'resource_fault',False)

    def _fail(self,job,stage):
        with self.core.transaction():
            if job['state'] not in ('running','cancel_requested'):raise ValueError('invalid_transition')
            job.update(state='cancelled' if job['state']=='cancel_requested' else 'failed',stage=stage)
            job['attempts'][-1]['reaped']=True;self.core._save(job)


def main():
    """Bounded installed diagnostic session; JSON lines, no public listener."""
    import sys
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--persistent-registration')
    parser.add_argument('--persistent-config')
    args=parser.parse_args()
    if bool(args.persistent_registration)!=bool(args.persistent_config):
        raise ValueError('explicit_persistent_configuration_required')
    configuration={}
    if args.persistent_config:
        from service.persistent_init import read_file
        configuration=json.loads(read_file(Path(args.persistent_config)))
        if not isinstance(configuration,dict) or set(configuration)!={'profiles','budget'}:
            raise ValueError('closed_persistent_configuration')
    dispatcher=Dispatcher('/control/core',persistent_registration=args.persistent_registration,**configuration)
    try:
        for line in sys.stdin:
            value=json.loads(line)
            if set(value)-{'op','args','kwargs'}:raise ValueError('closed_command')
            try:
                result=dispatcher.call(value['op'],*value.get('args',[]),**value.get('kwargs',{}))
                print(json.dumps(dict(ok=True,value=result)),flush=True)
            except Exception as exc:
                print(json.dumps(dict(ok=False,error=type(exc).__name__)),flush=True)
    finally:dispatcher.close()


if __name__=='__main__':main()
