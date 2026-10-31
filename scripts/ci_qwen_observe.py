"""Bounded host timings only; no tensor reads, synchronization or generation edits."""
import time


def observe(model, emit, *, clock=time.monotonic):
    talker = model.model.talker
    tokenizer = model.model.speech_tokenizer
    original = tokenizer.decode
    summary = dict(talker_forward_started=0, talker_forward_completed=0)
    last_emit = [0.0]
    def publish(force=False):
        now = clock()
        if force or now-last_emit[0] >= 1:
            try:
                emit(**summary)
            except Exception as exc:
                # Observation cannot replace a model exception or alter outputs.
                # Retain the diagnostic cause for the next bounded publication.
                summary['observation_error'] = dict(type=type(exc).__name__, message=str(exc)[:1000])
            last_emit[0] = now
    def before(module, args):
        now = clock()
        summary['talker_forward_started'] += 1
        summary.setdefault('first_forward_started', now)
        summary['last_forward_started'] = now
        publish()
    def after(module, args, output):
        now = clock()
        summary['talker_forward_completed'] += 1
        summary.setdefault('first_forward_completed', now)
        summary['last_forward_completed'] = now
        publish()
    handles = [talker.register_forward_pre_hook(before), talker.register_forward_hook(after)]
    def decode(*args, **kwargs):
        summary['decode_started'] = clock()
        publish(True)
        try:
            value = original(*args, **kwargs)
        except BaseException:
            summary['decode_failed'] = True
            raise
        else:
            summary['decode_completed'] = clock()
            return value
        finally:
            publish(True)
    tokenizer.decode = decode
    def close():
        for handle in handles: handle.remove()
        tokenizer.decode = original
        publish(True)
    return close
