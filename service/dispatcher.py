"""Installed internal fixed synthetic dispatcher. R1/P1 activation stays closed.

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
from service.persistence import PublicationQuarantine
from service.admission import Admission
from service.dispatch_store import DispatchStore
from service.supervisor_runtime import SupervisorRuntime
from worker.lifecycle import State


class Dispatcher:
    def __init__(self, root, *, ipc='/ipc', staging='/staging', runtime=None, fault=lambda _:None, budget=None, profiles=None):
        self.root, self.staging = Path(root), Path(staging)
        self.runtime = runtime or SupervisorRuntime(ipc)
        self.fault = fault
        self.budget, self.current_profiles = budget, profiles
        self.commands=queue.Queue(64)
        self.initialized=Future()
        self.thread=threading.Thread(target=self._run,name='control-writer',daemon=True)
        self.thread.start()
        self.initialized.result(timeout=5)

    def call(self, op, *args, timeout=3, **kwargs):
        if op not in ('submit','job','jobs','events','artifact','asset','cancel','status','replace','close'):
            raise ValueError('unknown_control_command')
        future=Future()
        self.commands.put_nowait((op,copy.deepcopy(args),copy.deepcopy(kwargs),future))
        return future.result(timeout=timeout)

    def close(self):
        self.call('close');self.thread.join(timeout=4)
        if self.thread.is_alive():raise RuntimeError('control_close_timeout')

    def _run(self):
        core=None
        try:
            core=self.core=Core(self.root)
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
            with core.transaction():
                for reservation in self.admission.store.rows():
                    if not self.persistence_fault and reservation['state']=='queued' and core.job(reservation['id'])['state']=='cancelled':
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
            if not self.initialized.done():self.initialized.set_exception(exc)
            else:self.error=type(exc).__name__
        finally:
            if hasattr(self,'pool'):self.pool.shutdown(wait=True,cancel_futures=True)
            if core:core.close()

    def _command(self,op,*args,**kwargs):
        if op=='close':self.stopping=True;return None
        if op=='status':
            row=self.store.current()
            return dict(reconciled=self.reconciled,quarantined=self.quarantined,
                phase=row['phase'] if row else 'idle',runtime_id=row['runtime'] if row else None,
                resource_activation=False,persistent_activation=False,error=self.error,
                resource_fault=self.resource_fault,persistence_fault=self.persistence_fault,
                stop_obligation=self.stop_obligation,
                reservations=self.admission.store.rows())
        if op=='cancel':
            job=self.core.request_cancel(*args,**kwargs,_cancel_unstarted=True,_admission=self.admission)
            self.next_poll=0
            return job
        if op in ('submit','replace'):
            # Idempotent lookup is a read, including during quarantine/reconnect.
            if op=='submit':
                if len(args)!=2 or not set(kwargs)<= {'retry_of','fixture','seconds'}:raise ValueError('closed_synthetic_command')
                existing=self.core.db.execute("SELECT job FROM jobs WHERE principal=? AND operation='speech_generate' AND key=?",args).fetchone()
                if existing:return json.loads(existing[0])
            if not self.reconciled or self.quarantined or self.resource_fault:raise ValueError('startup_or_quarantine')
            if op=='replace':
                if self.pending or self.store.current():raise ValueError('lane_busy')
                job_id=args[0]
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
                if self.persistence_fault:return
                self._launch(row,'ack',lambda:self.runtime.call('ack',dict(runtime_id=row['runtime'])));return
            if row['phase']=='intent':
                # Only a new live allocation reaches prepare. Startup never replays intent.
                self.store.update(row,phase='unknown');self.quarantined=True;self.reconciled=True
            if row['runtime'] is None:
                self.quarantined=True;self.reconciled=True;return
            self._launch(row,'observe',lambda:self.runtime.observe(row['runtime'],1.5));return
        if not self.reconciled:
            # Validate actual installed supervisor before admitting anything.
            if not hasattr(self,'hello'):
                self.hello=self.pool.submit(self.runtime.call,'hello',{})
            if not self.hello.done():return
            try: value=self.hello.result();self.quarantined=self.orphaned or self.resource_fault or bool(value['quarantined'])
            except Exception:self.quarantined=True
            self.reconciled=True;return
        if self.quarantined:return
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
    dispatcher=Dispatcher('/control/core')
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
