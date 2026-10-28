"""CPU family policy on the existing native owned-exec/journal lifecycle."""
import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal

from worker import native_cpu_protocol as p
from worker.native_supervisor import NativeSupervisor, Journal, private
from worker.native_cpu_availability import Availability, validate as available
from worker.engines.native_kokoro import STARTUP_SECONDS, JOB_SECONDS
from worker.native_cpu_resources import assess
from worker.transport import Refused,canonical


class CPUJournal(Journal):
    protocol=p
    version=p.JOURNAL_VERSION

    def __init__(self,root,*args):
        if (Path(root)/'owner-failure.json').exists() or (Path(root)/'owner-failure.json').is_symlink():raise Refused('native_cpu_retained_owner_failure')
        if (Path(root)/'predecessor').exists():raise Refused('native_cpu_renewal_unsupported')
        super().__init__(root,*args)


class NativeCPUSupervisor(NativeSupervisor):
    protocol=p
    journal_type=CPUJournal
    child_filename='native_cpu_child.py'
    receipt_version=p.RECEIPT_VERSION
    startup_seconds=STARTUP_SECONDS
    job_seconds=JOB_SECONDS
    disk_reserve_bytes=1024**3

    def __init__(self,*args,sampler_factory=None,qualification=None,_boot=None,**kwargs):
        from worker.native_cpu_qualification import Session
        if qualification is not None and type(qualification) is not Session:raise Refused('native_cpu_qualification_authority')
        self.qualification=qualification
        if _boot is not None:p._hex(_boot,32)
        self._configured_boot=_boot
        if sampler_factory is None:
            from scripts.ci_process_rss import Sampler
            sampler_factory=Sampler
        self.sampler_factory=sampler_factory
        if 'identity' not in kwargs:
            from scripts.ci_process_rss import ProcessRSS
            kwargs['identity']=lambda handle:str(ProcessRSS(handle.pid).sample()['identity'])
        super().__init__(*args,**kwargs)

    def new_boot(self):
        return super().new_boot() if self._configured_boot is None else self._configured_boot

    def owner_failure(self,exc):
        super().owner_failure(exc)
        self.owner_error=dict(type=type(exc).__name__,message=str(exc)[:1000],
            cause_type=type(exc.__cause__).__name__ if exc.__cause__ else None,
            cause=str(exc.__cause__)[:1000] if exc.__cause__ else None)
        # A diagnostic write cannot delay a live model's independent kill path.
        # After confirmed wait, retain the first owner validation/storage cause.
        if self.active is not None and self.active.get('waited'):
            try:
                fd=os.open(self.journal.root/'owner-failure.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd,'wb') as stream:stream.write(canonical(self.owner_error));stream.flush();os.fsync(stream.fileno())
            except Exception as failure:self.owner_error['write_error']=dict(type=type(failure).__name__,message=str(failure)[:1000])

    def dispatch(self,request):
        if self.qualification is None:return super().dispatch(request)
        with self.qualification.scope():return super().dispatch(request)

    def prepare(self,q):
        from worker.native_cpu_qualification import MAX_JOBS
        if self.qualification is not None and not any(r['binding']==q['binding'] for r in self.journal.value['records']) and len(self.journal.value['records'])>=MAX_JOBS:
            raise Refused('native_cpu_qualification_job_limit')
        return super().prepare(q)

    def make_availability(self,registrations,manifest,provider,utc,clock):
        inventories=[r['inventory'] for r in registrations.values()]
        if any(canonical(p.inventory_wire(i))!=canonical(p.inventory_wire(inventories[0])) for i in inventories):
            raise Refused('native_cpu_inventory')
        return Availability(inventories[0],manifest,provider=provider,utc=utc,clock=clock)

    def validate_output(self,snapshot,proposal,data,execution):
        from worker.native_child import validate_proposal
        return validate_proposal(snapshot,proposal,data,execution,protocol=p)

    def registration(self,r,verify=False,*,candidate=None,current=True):
        from contracts import validate as oracle
        reg=candidate if candidate is not None else self.registrations.get(r['binding'])
        if reg is None or canonical(reg['snapshot'])!=canonical(r['snapshot']):raise Refused('native_cpu_registration')
        job,context,assets,profiles=r['snapshot']
        if self.qualification is not None and not current:
            oracle.stored_context(job,context)
            selected=next(p for p in profiles if p['id']=='kokoro-native-cpu')
            self.qualification.authorize(context,selected,job,live=False)
        else:oracle.admission(job,context,profiles)
        attempt=oracle.active_attempt(job)
        if job['state'] not in ('running','cancel_requested') or attempt['reaped'] or assets:
            raise Refused('native_cpu_attempt')
        spec=p.native_execution(context)
        if reg['execution']!=spec or set(reg['bundles'])!={spec.bundle_id}:raise Refused('native_cpu_registration')
        binding=p.native_binding(context)
        if (binding['deployment_id'],binding['configuration_sha256'],binding['lane'],binding['runtime_manifest_sha256'])!=(self.journal.deployment,self.journal.configuration,self.journal.value['lane'],self.manifest):
            raise Refused('native_cpu_registration')
        if current:
            if binding['boot']!=self.boot:raise Refused('native_cpu_registration')
            available(self.availability.get(),reg['inventory'],self.manifest,now=self.utc())
        if verify:
            from service.execution_context import verify_local
            verify_local(context,reg['bundles'])
        return reg

    def register_snapshot(self,snapshot):
        from service.worker_transport import Authority
        from lib.spoken_text import freeze
        from service.native_renewal import host_registrations,capacity,publish
        authority=Authority(*snapshot);context=authority.context;spec=p.native_execution(context)
        matches=[]
        for template in self._preset_templates.values():
            if template['snapshot'][0]['principal_id']!=authority.job['principal_id']:continue
            try:
                derived=p.owned_preset_context(template['snapshot'][1],authority.job['principal_id'],spec.preset,spec.voice_id)
                derived['plan']['request']=copy.deepcopy(context['plan']['request']);freeze(derived['plan'])
                if canonical(derived)!=canonical(context):continue
                reg=copy.deepcopy(template);reg.update(snapshot=json.loads(authority._snapshot),execution=spec)
                matches.append(reg)
            except (Refused,ValueError,KeyError):continue
        if len(matches)!=1:raise Refused('native_cpu_registration')
        reg=matches[0];binding=authority.binding
        self.registration(dict(binding=binding,snapshot=reg['snapshot']),candidate=reg)
        if binding in self.registrations:
            if self.registrations[binding]!=reg:raise Refused('native_cpu_registration')
            return binding
        capacity('registrations',len(set(self.registrations)|set(host_registrations(self.journal.root))),63)
        publish(self.journal.root/('registration-'+binding),snapshot,cap=48000)
        self.registrations[binding]=reg
        return binding

    def validate_output(self,snapshot,proposal,data,execution):
        from worker.native_child import validate_proposal
        return validate_proposal(snapshot,proposal,data,execution,protocol=p)

    def exchange(self,connection,config,cutoff,on_loaded):
        from worker.native_cpu_child import parent_exchange,ChildFailure
        def loaded(value):
            on_loaded(value)
            with self.handle_lock:
                if self.active is not None:self.active['cpu_loaded']=True
        try:return parent_exchange(connection,config,cutoff,on_loaded=loaded)
        except ChildFailure as exc:
            with self.handle_lock:
                if self.active is not None:self.active['child_error']=exc.error
            raise

    def child_environment(self,r):
        root=self.directory(r)/'tmp';root.mkdir(mode=0o700)
        # Initialize before Popen's result can become visible to the watchdog.
        # The actual sampler starts once the owner has the returned handle.
        self.initialize_sampling(self.active)
        return dict(PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',HOME=str(root),TMPDIR=str(root),
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')

    def initialize_sampling(self,a):
        a['rss_started']=self.clock();a['rss_exit_observed']=None
        a['sampler']=None;a['sampler_started']=False;a['cpu_signals']=[];a['cpu_term_attempted']=False;a['cpu_kill_attempted']=False

    def model_started(self,a,r):
        try:
            a['sampler']=self.sampler_factory(a['handle'].pid);a['sampler'].start();a['sampler_started']=True
        except Exception as exc:
            a['sampler_start_error']=dict(type=type(exc).__name__,message=str(exc)[:1000],errno=getattr(exc,'errno',None),observed_at=self.clock())
            a['cancel']=True;a['resource_stop_reason']='native_cpu_rss_lost'

    def signal_tick(self):
        # Same exclusive Popen custody and fixed TERM/KILL grace as the shared
        # lifecycle. ProcessGone does not create death proof or renew freshness.
        with self.handle_lock:
            a=self.active
            if a is None or a['handle'] is None:return
            h=a['handle'];sampler=a.get('sampler')
            state=sampler.snapshot() if a.get('sampler_started') else dict(last=None,peak=0,count=0,error=a.get('sampler_start_error'))
            now=self.clock()
            if a['exit'] is None and (state['count'] or state['error']):
                status=h.poll()
                if status is not None:a['exit']=status;a['rss_exit_observed']=self.clock()
            result=assess(state,started=a['rss_started'],now=self.clock(),exit_observed_at=a['rss_exit_observed'])
            a['resources']=result
            if result['failure']:
                a['cancel']=True
                a.setdefault('resource_stop_reason',result['failure'])
                if result['failure']=='native_cpu_identity_changed':self.failed=True;return
            if a['exit'] is not None:
                if sampler is not None:sampler.request_stop()
                return
            if not (a['cancel'] or now>=self.cutoffs[a['id']]):return
            a.setdefault('cpu_stop_requested_at',now)
            a.setdefault('cpu_stop_reason',a.get('resource_stop_reason') or ('cancelled' if a['cancel'] else 'request_deadline' if a.get('cpu_loaded') else 'startup_deadline'))
            if a.get('cpu_signal_blocked'):return

            if now-a['cpu_stop_requested_at']>=1.2:
                self.failed=True
                return
            # Stop accepting late samples before a no-sample cancellation can
            # release its PID; an unreaped, exclusively owned child cannot reuse.
            if not state['count'] and sampler is not None:sampler.request_stop()
            try:
                status=h.poll()
                if status is not None:
                    a['exit']=status;a['rss_exit_observed']=self.clock()
                    if sampler is not None:sampler.request_stop()
                    return
                try:
                    if self.identity(h)!=a['start_identity']:raise Refused('native_identity')
                except ProcessLookupError:
                    # A process can disappear before waitpid exposes exit. Probe
                    # existence without signalling; never signal an absent PID.
                    try:os.kill(h.pid,0)
                    except ProcessLookupError:return
                number=signal.SIGTERM if not a['cpu_term_attempted'] else signal.SIGKILL if a['term'] is not None and now-a['term']>=.2 and not a['cpu_kill_attempted'] else None
                if number is not None:
                    a['cpu_term_attempted' if number==signal.SIGTERM else 'cpu_kill_attempted']=True
                    item=dict(signal=number.name,requested_at=self.clock(),accepted=False,error=None);a['cpu_signals'].append(item)
                    try:self.signal_group(h.pid,number);item['accepted']=True
                    except ProcessLookupError as exc:item['error']=dict(type=type(exc).__name__,errno=exc.errno);return
                    if number==signal.SIGTERM:a['term']=self.clock()
                    else:a['killed']=self.clock()
                elif a['killed'] is not None and now-a['killed']>=1:self.failed=True
            except Exception as exc:
                a['cpu_signal_error']=dict(type=type(exc).__name__,message=str(exc)[:1000],errno=getattr(exc,'errno',None));a['cpu_signal_blocked']=True;self.failed=True
                if a['cpu_signals'] and not a['cpu_signals'][-1]['accepted'] and a['cpu_signals'][-1]['error'] is None:
                    a['cpu_signals'][-1]['error']=dict(type=type(exc).__name__,errno=getattr(exc,'errno',None))

    def terminal(self,r,processes,timely,proposal=None,output=None):
        if self.failed or r['id'] in self.uncertain:raise Refused('native_unknown')
        a=self.active if self.active is not None and self.active['id']==r['id'] else None
        state=dict(last=None,error=None,peak=0,count=0)
        error=None
        if a is not None:
            sampler=a.get('sampler')
            if a.get('sampler_started'):
                if not sampler.stop():raise Refused('native_cpu_sampler_cleanup')
                state=sampler.snapshot()
            else:
                if sampler is not None and sampler.thread.is_alive():raise Refused('native_cpu_sampler_cleanup')
                state=dict(last=None,peak=0,count=0,error=a.get('sampler_start_error'))
            resources=assess(state,started=a['rss_started'],now=self.clock(),exit_observed_at=a['rss_exit_observed'])
            # Once lost, a later exit must not convert the observation to success.
            if a.get('resource_stop_reason'):
                resources['failure']=a['resource_stop_reason']
                if resources['failure']=='native_cpu_rss_lost':resources['observation_lost']=True
            error=a.get('child_error')
            if error is None and a.get('receiver_error'):
                error=dict(code='native_cpu_worker_failed',type=a['receiver_error']['type'],message=a['receiver_error']['message'],cause_type=None,cause=None)
            temporary=self.directory(r)/'tmp'
            if temporary.exists():
                private(temporary,True)
                for path in temporary.rglob('*'):
                    info=path.lstat()
                    if path.is_symlink() or info.st_uid!=os.getuid() or path.is_file() and info.st_nlink!=1:raise Refused('native_cpu_temporary')
                shutil.rmtree(temporary)
        else:resources=assess(state,started=self.clock(),now=self.clock())
        resources['thread_stopped']=True
        if resources['failure'] or error:timely=False;proposal=output=None
        receipt=dict(self.journal.expected(r),version=self.receipt_version,reaped=True,timely=timely,
            reap_method='darwin-owned-exec-v1',processes=processes,receiver_closed=True,resources=resources,error=error,termination=dict(reason=None if a is None else a.get('cpu_stop_reason'),
                signals=[] if a is None else a['cpu_signals'],signal_error=None if a is None else a.get('cpu_signal_error'),
                observed_exit_at=None if a is None else a['rss_exit_observed']))
        p.receipt(receipt,expected=self.journal.expected(r))
        self.persist(r,phase='terminal',receipt=receipt,proposal=proposal,output=output)
