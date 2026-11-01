"""Trusted frozen-family routing. Never choose a validator from received bytes."""
from worker import native_protocol as qwen
from worker import native_cpu_protocol as cpu


def protocol(context):
    return cpu if cpu.frozen_native(context) else qwen


def frozen_native(context):
    return cpu.frozen_native(context) or qwen.frozen_native(context)


def terminal_receipt(value, *, authority, runtime_id, local_import_uid=None):
    return protocol(authority.context).terminal_receipt(value,authority=authority,
        runtime_id=runtime_id,local_import_uid=local_import_uid)
