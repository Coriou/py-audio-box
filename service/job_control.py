"""Finite durable cursors and bounded waiting outside the dispatcher owner."""
import asyncio
import hashlib
import hmac
import time
from contracts.validate import validate
from service.catalogue import Failure


def cursor_encode(seq, principal, kind, job_id):
    binding = hashlib.sha256((principal+'\0'+kind+'\0'+job_id).encode()).hexdigest()[:24]
    return binding+':'+str(seq)


def cursor_decode(cursor, principal, kind, job_id):
    if cursor == '': return 0
    try:
        binding, number = cursor.split(':')
        if not number.isascii() or not number.isdecimal() or not 0 < int(number) <= 9007199254740991:
            raise ValueError()
        seq = int(number)
        if not hmac.compare_digest(cursor_encode(seq,principal,kind,job_id),cursor): raise ValueError()
        return seq
    except (ValueError, AttributeError):
        raise ValueError('invalid_request') from None


async def wait_job(dispatcher, principal, payload, disconnected):
    """Each owner call is a detached read. Disconnect has no job side effect."""
    from service.domain import production_call
    validate('job_wait_request',payload)
    deadline = time.monotonic()+payload['wait_seconds']
    while True:
        result = await asyncio.to_thread(production_call, dispatcher, principal, 'job_events',
                                         dict(job_id=payload['job_id'],cursor=payload['cursor'],limit=20))
        expired = time.monotonic() >= deadline
        if result['events'] or result['job']['state'] in ('succeeded','failed','cancelled') or expired:
            return validate('job_wait_result',dict(result,deadline_reached=expired))
        if await disconnected():
            raise Failure('not_ready','Wait disconnected; job continues.',action='Resume with the last returned cursor.')
        await asyncio.sleep(min(.05,max(0,deadline-time.monotonic())))
