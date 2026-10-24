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
    def __init__(self, root, *, ipc='/ipc', staging='/staging', runtime=None, fault=lambda _:None, budget=None, profiles=None, persistent_registration=None, production_contexts=None, installed_activation=None):
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
        self.thread=threading.Thread(target=self._run,name='control-writer',daemon=True)
        self.thread.start()
        self.initialized.result(timeout=15 if installed_activation is not None else 5)

    def call(self, op, *args, timeout=3, **kwargs):
        if op not in ('submit','job','jobs','events','artifact','asset','cancel','status','replace','inventory_page','reconcile_startup','close','voice_list','voice_get','voice_version','voice_resolve','voice_summaries','voice_version_summaries','asset_ingest','asset_metadata','asset_content','production_prepare','production_generate','production_save','production_results','production_job','production_cancel','production_artifact'):
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
            from service.recovery import startup_read, check_completion_hello, _startup_read_lock
            # Hold the old owner's exclusion through all private readers. Release
            # before Core's independent acquisition (never double-acquire flock).
            # A competing owner in that gap either makes Core refuse, or completes
            # before Core repeats the deployment/completion/hello binding below.
            if self.installed_activation is not None:
                from service.entrypoint import InstalledActivation
                if type(self.installed_activation) is not InstalledActivation:
                    raise ValueError('checked_installed_activation_required')
                self.installed_activation.precheck(self)
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
            core=self.core=Core(self.root)
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
            self.admission=Admission(core,self.staging,self.budget)
            self.store=DispatchStore(core,self.admission)
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
            with core.transaction():
                for reservation in self.admission.store.rows():
                    if self._activation_open() and not self.orphaned and not self.resource_fault and not self.integrity_fault and not self.persistence_fault and reservation['state']=='queued' and core.job(reservation['id'])['state']=='cancelled':
                        self.admission.store.release_tx(reservation['id'],unstarted=True)
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
                    if startup_error is not None:self.initialized.set_exception(startup_error)

    def _activation_open(self):
        # One closed predicate for commands, scheduler and queued release.
        return not self.enrolled or self.installed_checked

    def _ready(self):
        return (self._activation_open() and self.reconciled and not self.closing and
                not self.orphaned and not self.quarantined and not self.resource_fault and
                not self.persistence_fault and not self.integrity_fault and
                (not self.installed_checked or all(
                    p['support']=='qualified' and p['installation']=='installed' and
                    p['readiness']=='ready' and p['verification']=='measured'
                    for p in self.profiles)))

    def _production_command(self, op, principal, payload, *, key=None):
        from contracts import validate as oracle
        from service.production import Production
        from service.catalogue import OPERATIONS
        from service.voice_store import _id
        operation = {'production_prepare':'voice_prepare', 'production_generate':'speech_generate',
                     'production_save':'voice_save', 'production_results':'voice_results',
                     'production_job':'job_get', 'production_cancel':'job_cancel',
                     'production_artifact':'artifact_get'}[op]
        oracle.validate(OPERATIONS[operation]['request'], payload)
        if OPERATIONS[operation]['idempotency']: _id(key)
        domain = Production(self.core)
        def owned(job_id):
            job = self.core.job(job_id)
            if job['principal_id'] != principal: raise ValueError('not_found')
            return job
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
        if op == 'production_save':
            if self.installed_checked and not self._ready(): raise ValueError('not_ready')
            if not self._activation_open() or self.closing or self.integrity_fault or self.persistence_fault: raise ValueError('not_ready')
            owned(payload['job_id'])
            return domain.save(principal, key, payload['job_id'], payload['candidate_id'], payload['voice_id'], expected_revision=payload['expected_revision'])
        # Duplicate lookup retains the original frozen identity and reservation.
        row = self.core.db.execute('SELECT fingerprint,job FROM jobs WHERE principal=? AND operation=? AND key=?', (principal,operation,key)).fetchone()
        if row:
            oracle.require(row[0] == oracle.fingerprint(principal,operation,payload), 'idempotency_conflict')
            job = json.loads(row[1])
        else:
            if not self._activation_open() or self.closing or not self.reconciled or self.quarantined or self.resource_fault or self.integrity_fault:
                raise ValueError('not_ready')
            if self.production_contexts is None: raise ValueError('not_ready')
            context = self.production_contexts.resolve(principal, operation, payload)
            method = domain.prepare if op == 'production_prepare' else domain.generate
            job = method(principal, key, context, _admission=self.admission,
                         _profiles=self.profiles, _dispatch_settings=('silence',8))
        return dict(job_id=job['id'], input_sha256=job['input_sha256'], state='queued', status_url='/v1/jobs/'+job['id'])

    def _command(self,op,*args,**kwargs):
        if op.startswith('production_'): return self._production_command(op,*args,**kwargs)
        if op=='asset_ingest':
            if self.installed_checked and not self._ready(): raise ValueError('not_ready')
            from service.ingest import ingest_local
            if len(args)!=2 or type(args[1]) is not bytes or kwargs: raise ValueError('invalid_request')
            return ingest_local(self.core,*args)
        if op in ('asset_metadata','asset_content'):
            from service.domain import owned_asset
            return owned_asset(self.core,*args,content=op=='asset_content',**kwargs)
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
            return dict(reconciled=self.reconciled,quarantined=self.quarantined,
                phase=row['phase'] if row else 'idle',runtime_id=row['runtime'] if row else None,
                resource_activation=self._ready(),persistent_activation=self.enrolled and self._ready(),error=self.error,
                resource_fault=self.resource_fault,persistence_fault=self.persistence_fault,
                stop_obligation=self.stop_obligation,
                reservations=self.admission.store.rows(), integrity=self.integrity,
                integrity_fault=self.integrity_fault)
        if op=='cancel':
            if not self._activation_open(): raise ValueError('persistent_activation_closed_until_C1')
            if self.installed_checked and self.core.job(args[0])['state']=='queued' and not self._ready(): raise ValueError('not_ready')
            if self.integrity_fault and self.core.job(args[0])['state']=='queued':raise ValueError('integrity_blocks_release')
            job=self.core.request_cancel(*args,**kwargs,_cancel_unstarted=True,_admission=self.admission)
            self.next_poll=0
            return job
        if op in ('submit','replace'):
            # Idempotent lookup is a read, including during quarantine/reconnect.
            if op=='submit':
                if len(args)!=2 or not set(kwargs)<= {'retry_of','fixture','seconds'}:raise ValueError('closed_synthetic_command')
                existing=self.core.db.execute("SELECT job FROM jobs WHERE principal=? AND operation='speech_generate' AND key=?",args).fetchone()
                if existing:return json.loads(existing[0])
            if not self._activation_open(): raise ValueError('persistent_activation_closed_until_C1')
            if self.installed_checked: raise ValueError('installed_production_commands_required')
            if self.closing or not self.reconciled or self.quarantined or self.resource_fault or self.integrity_fault:raise ValueError('startup_or_quarantine')
            if op=='replace':
                if self.pending or self.store.current():raise ValueError('lane_busy')
                job_id=args[0]
                check_operator_resolved(self.core.db,job_id)
                row=self.store.allocate(self.profiles,'silence',8,replacement=job_id)
                self.fault('prespawn')
                self._launch(row,'prepare',lambda:self.runtime.prepare(row['authority'],row['fixture']))
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

    def _launch(self,row,op,fn):
        command=self.store.command(row,op)
        self.fault('before_'+op)
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
        if (self.stop_obligation and row and row['runtime'] and not row['outcome']
                and not row['receipt'] and row['runtime'] not in self.resource_stop_sent and not self.pending):
            self._launch(row,'stop',lambda:self.runtime.terminate(row['runtime'],1.5))
            self.resource_stop_sent.add(row['runtime'])
            return True
        return False

    def _tick(self):
        if not self._activation_open(): return  # Legacy enrolled remains inspect-only.
        if getattr(self.core,'_publication_storage_failed',False):
            self.persistence_fault=self.quarantined=True
            return  # Exact reap preceded publication; storage needs reopen, no live retry.
        self._resource_check()
        if self.pending:
            future,row,op,command=self.pending
            if not future.done():return
            self.pending=None
            try:value=future.result()
            except Exception:
                self.store.complete(command,False)
                if op=='ack':self.next_poll=time.monotonic()+.2;return
                self.store.update(row,phase='unknown');self.quarantined=True
                self.reconciled=True;self.next_poll=time.monotonic()+.2;return
            self.fault('after_'+op)
            self.store.complete(command,True)
            if op=='prepare':
                self.fault('preregistration')
                self.store.update(row,runtime=value,phase='bound')
                self.fault('prestart')
                bound=self.store.current()
                if self._service_stop(bound):return
                self._launch(bound,'start',lambda:self.runtime.start(value));return
            if op=='start':self.store.update(row,phase='observing');self.fault('postspawn')
            elif op=='observe':
                if value.runtime_id!=row['runtime']:raise ValueError('unbound_observation')
                self.reconciled=True
                self.quarantined=value.state==State.UNKNOWN or self.resource_fault
                if value.state==State.STOPPED:
                    if self.persistence_fault and row['receipt']:
                        if self.core.job(row['job_id'])['state'] not in ('succeeded','failed','cancelled'):
                            self._terminal(row,row['receipt'],None)
                    else:
                        self.fault('observation_reap')
                        self._launch(row,'collect',lambda:self.runtime.collect(row['authority'],row['runtime']));return
                self.store.update(row,phase='unknown' if self.quarantined else 'observing')
                if value.state==State.ALIVE and not self.stop_obligation and self.core.job(row['job_id'])['state']=='cancel_requested':
                    self._launch(row,'stop',lambda:self.runtime.terminate(row['runtime'],1.5));return
            elif op=='collect':
                if value is None:raise ValueError('terminal_receipt_missing')
                self._terminal(row,*value)
            elif op=='ack':
                if value!={'acked':True}:raise ValueError('ack_required')
                self.store.update(row,acked=1,phase='acknowledged');self.quarantined=getattr(self,'resource_fault',False)
            self.next_poll=time.monotonic()+.03
        if self.pending:return
        if self._service_stop(self.store.current()):return
        if time.monotonic()<self.next_poll:return
        row=self.store.current()
        if row:
            if row['outcome']:
                if self.integrity_fault:return
                if self.persistence_fault:return
                self._launch(row,'ack',lambda:self.runtime.call('ack',dict(runtime_id=row['runtime'])));return
            if row['phase']=='intent':
                # Only a new live allocation reaches prepare. Startup never replays intent.
                self.store.update(row,phase='unknown');self.quarantined=True;self.reconciled=True
            if row['runtime'] is None:
                self.quarantined=True;self.reconciled=True;return
            self._launch(row,'observe',lambda:self.runtime.observe(row['runtime'],1.5));return
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
            self.reconciled=True;return
        if self.quarantined or self.integrity_fault:return
        queued=next((json.loads(r[0]) for r in self.core.db.execute('SELECT job FROM jobs ORDER BY seq') if json.loads(r[0])['state']=='queued'),None)
        if queued:
            settings=self.core.db.execute('SELECT fixture,seconds FROM dispatch_settings WHERE job_id=?',(queued['id'],)).fetchone()
            if settings is None:raise ValueError('fixed_dispatch_settings_required')
            row=self.store.allocate(self.profiles,*settings)
            self.fault('prespawn')
            self._launch(row,'prepare',lambda:self.runtime.prepare(row['authority'],row['fixture']))

    def _terminal(self,row,receipt,observation):
        if getattr(self.core,'_publication_storage_failed',False):
            self.persistence_fault=self.quarantined=True
            return  # Only reopen may resolve publication storage ambiguity.
        authority=row['authority'];job=self.core.job(row['job_id'])
        if (receipt['runtime_id']!=row['runtime'] or receipt['binding']!=authority.binding or
            receipt['reaped'] is not True or receipt['reap_method']!='pid1-waitpid-echild' or
            receipt['original_deadline']!=authority.job['deadline']):raise ValueError('unbound_receipt')
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
                self._resource_check(publication=True)
                if getattr(self,'resource_fault',False):raise ValueError('resource_violation')
                if intent:
                    if receipt['timely'] is not True:raise ValueError('late_publication_intent')
                    stored=json.loads(self.core.db.execute('SELECT artifact FROM publications WHERE job_id=?',(job['id'],)).fetchone()[0])
                    for key in ('generation','attempt_id','owner_id','staging_id'):
                        expected=authority.job['generation'] if key=='generation' else authority.job['attempts'][-1]['id' if key=='attempt_id' else key]
                        if stored[key]!=expected:raise ValueError('stale_publication_intent')
                    self.core._commit_publication(job['id'])
                elif receipt['timely'] is True and observation is not None:
                    self.core._publish(job['id'],dict(message=observation.message,manifest=observation.manifest),
                        fault=self.fault,stage_root=self.staging,engine_uid=receipt['engine_uid'])
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
