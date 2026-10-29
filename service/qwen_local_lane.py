"""Explicit, bounded local Qwen execution before runtime qualification."""

REVISION = 'qwen-local-unqualified-v1'
MEMORY_BYTES = 4 * 1024**3
DISK_BYTES = 10 * 1024**3
DEADLINE_SECONDS = 600


def profile_allowed(profile):
    return (profile['id'] == 'qwen-cpu' and profile['config_revision'] == REVISION
            and profile['device'] == 'cpu' and profile['topology'] == 'networkless_container'
            and profile['support'] == 'unqualified' and profile['installation'] == 'installed'
            and profile['configuration'] == 'valid' and profile['readiness'] == 'not_ready'
            and profile['verification'] == 'synthetic' and profile['qualification_receipt'] == ''
            and 0 < profile['limits']['ram_bytes'] <= MEMORY_BYTES
            and 0 < profile['limits']['disk_bytes'] <= DISK_BYTES
            and 0 < profile['limits']['deadline_seconds'] <= DEADLINE_SECONDS
            and profile['limits']['expensive_slots'] == 1
            and profile['limits']['resident_models'] == 1)


def context_allowed(context, profile):
    from worker.runtime_protocol import execution, resident_qwen_model
    spec = execution(context)
    return (profile_allowed(profile) and spec is not None and
            spec.engine == 'qwen' and spec.device == 'cpu' and
            resident_qwen_model(spec) in ('clone','Ryan') and
            spec.memory_bytes <= profile['limits']['ram_bytes'] and
            spec.timeout_s <= profile['limits']['deadline_seconds'])
