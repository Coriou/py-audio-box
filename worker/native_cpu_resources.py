"""CPU owned-child RSS observation policy using the existing in-process sampler.

A sampled limit is not a continuous OS memory bound. Observation loss fails
closed. ProcessGone is provisional only within the existing 0.5 s freshness.
"""
import errno
import math
from worker.engines.native_kokoro import MEMORY_BYTES
from worker.transport import Refused

STALE_SECONDS=.5


def assess(state,*,started,now,exit_observed_at=None):
    """Pure decision; never signals, polls, reaps or renews an elapsed deadline."""
    last=state['last'];error=state['error'];at=now if exit_observed_at is None else exit_observed_at
    # The real sampler has peak=None before its first observation. This conveys
    # no bound and retains the original freshness deadline, never a safe peak.
    peak=0 if state['peak'] is None and state['count']==0 else state['peak']
    result=dict(limit_bytes=MEMORY_BYTES,sampled_peak_bytes=peak,samples=state['count'],
        rss_bound_proven=False,observation_lost=False,last_valid_sample=last,
        age_at_observed_exit_seconds=None if last is None else max(0,at-last['observed_at']),
        sampler_error=error,exit_resolution=None,failure=None)
    if error:
        if error['type']=='ProcessGone' and error.get('errno') in (errno.ESRCH,errno.ENOENT):
            if last is not None and at-last['observed_at']<STALE_SECONDS:
                result['exit_resolution']='provisional' if exit_observed_at is None else 'confirmed_within_freshness'
            else:
                result.update(observation_lost=True,failure='native_cpu_rss_lost',exit_resolution='unconfirmed_past_freshness')
        else:result.update(observation_lost=True,failure='native_cpu_identity_changed' if error['type']=='ProcessReused' else 'native_cpu_rss_lost')
    elif last is None:
        if at-started>=STALE_SECONDS or exit_observed_at is not None:result.update(observation_lost=True,failure='native_cpu_rss_lost')
    elif at-last['observed_at']>=STALE_SECONDS:
        result.update(observation_lost=True,failure='native_cpu_rss_lost')
    if peak>MEMORY_BYTES:result['failure']='native_cpu_memory_limit'
    return result


def validate(value,*,processes,timely):
    from worker.native_protocol import _closed, _int
    _closed(value,('limit_bytes','sampled_peak_bytes','samples','rss_bound_proven','observation_lost',
        'last_valid_sample','age_at_observed_exit_seconds','sampler_error','exit_resolution','failure','thread_stopped'))
    if value['limit_bytes']!=MEMORY_BYTES or value['rss_bound_proven'] is not False or value['thread_stopped'] is not True:
        raise Refused('native_cpu_resource_receipt')
    _int(value['sampled_peak_bytes'],0,2**53-1);_int(value['samples'],0,2**31-1)
    if type(value['observation_lost']) is not bool:raise Refused('native_cpu_resource_receipt')
    if value['failure'] not in (None,'native_cpu_identity_changed','native_cpu_rss_lost','native_cpu_memory_limit'):
        raise Refused('native_cpu_resource_receipt')
    if value['exit_resolution'] not in (None,'provisional','confirmed_within_freshness','unconfirmed_past_freshness'):
        raise Refused('native_cpu_resource_receipt')
    if value['sampler_error'] is not None:
        _closed(value['sampler_error'],('type','message','errno','observed_at'))
        if any(type(value['sampler_error'][k]) is not str or len(value['sampler_error'][k])>1000 for k in ('type','message')):
            raise Refused('native_cpu_resource_receipt')
        if value['sampler_error']['errno'] is not None and type(value['sampler_error']['errno']) is not int:
            raise Refused('native_cpu_resource_receipt')
        stamp=value['sampler_error']['observed_at']
        if type(stamp) not in (int,float) or not math.isfinite(stamp):raise Refused('native_cpu_resource_receipt')
    last=value['last_valid_sample']
    if last is not None:
        _closed(last,('pid','identity','rss_bytes','method','observed_at'))
        _int(last['pid'],1,2**31-1);_int(last['rss_bytes'],0,2**53-1)
        if last['method'] not in ('proc_pid_rusage_v0','proc_pid_stat'):raise Refused('native_cpu_resource_receipt')
        if type(last['identity']) is not int or last['identity']<0 or type(last['observed_at']) not in (int,float) or not math.isfinite(last['observed_at']):
            raise Refused('native_cpu_resource_receipt')
    age=value['age_at_observed_exit_seconds']
    if age is not None and (type(age) not in (int,float) or not math.isfinite(age) or age<0):raise Refused('native_cpu_resource_receipt')
    if not processes:
        if timely or value['samples'] or last is not None:raise Refused('native_cpu_resource_receipt')
    elif last is not None and (last['pid']!=processes[0]['pid'] or str(last['identity'])!=processes[0]['start_identity']):raise Refused('native_cpu_resource_receipt')
    if timely and (value['samples']<1 or last is None or age is None or age>=STALE_SECONDS or
            value['observation_lost'] or value['failure'] is not None or value['sampled_peak_bytes']>MEMORY_BYTES or value['exit_resolution']=='provisional'):
        raise Refused('native_cpu_resource_receipt')
    return value
