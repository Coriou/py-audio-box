"""Native inference function and permanent child-only accidental-spawn guard.

No executable entry/launcher is exposed until the native owner/receiver contract
is implemented. Calling infer does not establish process or terminal authority.
"""
import os
import sys

from worker.transport import Refused

_guard_installed = False


class SpawnRefused(RuntimeError):
    pass


def _refuse_spawn(*args, **kwargs):
    raise SpawnRefused('native_child_spawn_forbidden')


def _audit_spawn(event, args):
    if event in {'subprocess.Popen', 'os.fork', 'os.forkpty', 'os.posix_spawn',
                 'os.system', 'os.exec', 'os.spawn', 'pty.spawn'}:
        _refuse_spawn()


def install_spawn_guard():
    """Permanent, process-global: call only on entry to the owned exec child.

    Audit plus explicit hooks cover ordinary Python process creation, including
    multiprocessing/resource_tracker. This is not a native-code sandbox.
    """
    global _guard_installed
    if _guard_installed:
        return
    sys.addaudithook(_audit_spawn)
    import subprocess
    from multiprocessing.process import BaseProcess
    subprocess.Popen = _refuse_spawn
    BaseProcess.start = _refuse_spawn
    for name in ('fork', 'forkpty', 'posix_spawn', 'posix_spawnp', 'system',
                 'execl', 'execle', 'execlp', 'execlpe', 'execv', 'execve',
                 'execvp', 'execvpe', 'spawnl', 'spawnle', 'spawnlp', 'spawnlpe',
                 'spawnv', 'spawnve', 'spawnvp', 'spawnvpe'):
        if hasattr(os, name):
            setattr(os, name, _refuse_spawn)
    _guard_installed = True


def infer(execution, text, bundles, references, inventory, *,
          reference_temp_dir=None, managed_owner_uid=None):
    """Real input verifier/complete loader/encoder with a closed native backend.

    The future entry must install_spawn_guard before calling this function. No
    arbitrary backend factory or caller-selected method crosses this boundary.
    """
    from worker.engines.qwen_execution import current_native_hardware
    from worker.engines.qwen import verify_service_inputs, execute
    if not _guard_installed:
        raise Refused('native_spawn_guard_required')
    current_native_hardware(execution, inventory)
    if execution.timeout_s > 30 or execution.max_output_bytes > 1048576:
        raise Refused('native_execution_limit')
    verify_service_inputs(execution, bundles, references, inventory,
                          owner_uid=managed_owner_uid)
    from lib.tts_backends.qwen_native import NativeQwenBackend
    return execute(execution, text, bundles, references, inventory,
        reference_temp_dir=reference_temp_dir, managed_owner_uid=managed_owner_uid,
        backend_factory=NativeQwenBackend)
