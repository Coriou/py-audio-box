"""Native single-load-per-job budgets; original job deadline remains absolute."""
import math
from worker.transport import Refused

STARTUP_SECONDS = 120
REQUEST_SECONDS = 30
DIAGNOSTIC_REQUEST_SECONDS = 120
JOB_SECONDS = STARTUP_SECONDS + REQUEST_SECONDS


def validate_job_cutoff(cutoff, now):
    if type(cutoff) not in (int, float) or not math.isfinite(cutoff) or not 0 < cutoff-now <= JOB_SECONDS:
        raise Refused('native_deadline')


def startup_cutoff(cutoff, now):
    validate_job_cutoff(cutoff, now)
    return min(cutoff, now+STARTUP_SECONDS)


def request_cutoff(cutoff, now, seconds):
    if type(seconds) is not int or not 0 < seconds <= REQUEST_SECONDS:
        raise Refused('native_execution_limit')
    if type(cutoff) not in (int,float) or not math.isfinite(cutoff) or not 0 < cutoff-now <= seconds:
        raise Refused('native_deadline')
    return cutoff
