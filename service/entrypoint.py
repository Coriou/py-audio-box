"""Installed control's private launch notification boundary.

No listener or Docker authority lives here. The explicit ASGI factory owns
Core only after corroborating the trusted notification with SQLite/live identity.
"""
import argparse
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

from service import install_policy as ip
from service.core import encode
from service.persistent_init import digest

GATE_DIRECTORY = Path('/tmp/speech-box-activation')
GATE_NAME = 'gate.json'
GATE_LIMIT = 65536

# Strong process-lifetime responsibility, including failed/unknown close outcomes.
_INSTALLED_LIFECYCLES = {}


class _ThreadResult:
    """Thread-owned completion, independent of cancellable asyncio wrappers."""
    def __init__(self, loop, function, completed):
        from concurrent.futures import Future
        self.result = Future()
        def work():
            try:
                value = function()
            except BaseException as exc:
                self.result.set_exception(exc)
            else:
                self.result.set_result(value)
            finally:
                try:
                    loop.call_soon_threadsafe(completed)
                except RuntimeError:
                    # Closed loop retains the result and unknown obligation.
                    pass
        submitted = loop.run_in_executor(None, work)
        submitted.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        self.wrapper = loop.create_task(self.wait())
        self.wrapper.add_done_callback(lambda done: None if done.cancelled() else done.exception())

    async def wait(self):
        import asyncio
        return await asyncio.shield(asyncio.wrap_future(self.result))


def projection(value, *, family=None):
    from service import installed_pocket_daemon as pd
    from service import installed_qwen_daemon as qd
    if family is qd.FAMILY:
        from service.daemon_helper import gate_projection
        return gate_projection(value, family=family)
    if family is pd.FAMILY:
        from service.daemon_helper import gate_projection
        return gate_projection(value, family=family)
    if family is not None:
        from service.installed_export import FAMILY
        if family is not FAMILY: raise ValueError("explicit_export_family_required")
        from service.installed_export import LAUNCH, host_registration
        from service.recovery import closed
        closed(value, "version launch_policy registration_id registration configuration_digest enrollment_digest provenance_digest observation")
        if type(value["version"]) is not int or value["version"] != 2 or value["launch_policy"] != LAUNCH:
            raise ValueError("installed_gate_version")
        host_registration(value["registration"])
        if digest(value["registration"]) != value["registration_id"]:
            raise ValueError("installed_gate_registration")
        for key in ("configuration_digest", "enrollment_digest", "provenance_digest"):
            if type(value[key]) is not str or not re.fullmatch("[0-9a-f]{64}", value[key]):
                raise ValueError("installed_gate_digest")
        # This file parser checks shape only. Authority comes from the later
        # selected configuration/enrollment/Docker/hello brackets.
        from service import daemon_volumes as dv
        from service.host_install import _registration_hello
        observation=value['observation']
        closed(observation,'kind binding inspections local hello')
        from service.installed_export import EVIDENCE
        if observation['kind'] != EVIDENCE: raise ValueError('daemon_gate_evidence')
        closed(observation['local'],'control worker')
        closed(observation['inspections'],'control worker')
        for role in ('control','worker'): dv.local(observation['local'][role],role,family=family)
        hello=observation['hello']
        closed(hello,'lane boot quarantined launches export_seal')
        _registration_hello({k:v for k,v in hello.items() if k!='export_seal'},value['registration'])
        seal=hello['export_seal']
        closed(seal,'configuration_digest converter_digest worker_image_digest profile_digest principals_digest publication_format')
        if type(seal['publication_format']) is not int or seal['publication_format'] != 2:
            raise ValueError('installed_export_seal_mismatch')
        for key,item in seal.items():
            if key=='publication_format': continue
            pattern='sha256:[0-9a-f]{64}' if key=='worker_image_digest' else '[0-9a-f]{64}'
            if type(item) is not str or not re.fullmatch(pattern,item):
                raise ValueError('installed_export_seal_mismatch')
        return value
    from service.recovery import closed, validate_registration
    if value.get('launch_policy') == 'installed-daemon-volumes-v1':
        from service.daemon_helper import gate_projection
        return gate_projection(value)
    closed(value, 'version registration_id registration configuration_digest enrollment_digest provenance_digest processes')
    if type(value['version']) is not int or value['version'] != 1:
        raise ValueError('installed_gate_version')
    if len(encode(value).encode()) > 16384:
        raise ValueError('installed_gate_budget')
    for key in ('registration_id', 'configuration_digest', 'enrollment_digest', 'provenance_digest'):
        if not isinstance(value[key], str) or not re.fullmatch('[0-9a-f]{64}', value[key]):
            raise ValueError('installed_gate_digest')
    record = value['registration']; validate_registration(record)
    if (digest(record) != value['registration_id'] or record['evidence'] != 'actual-host-inspection' or
            len(record['containers']) != 2):
        raise ValueError('installed_gate_registration')
    if type(value['processes']) is not list or len(value['processes']) != 2:
        raise ValueError('installed_gate_processes')
    seen = set(); namespaces = set()
    owners = {c['id']: c for c in record['containers']}
    if {c['role'] for c in owners.values()} != {'control', 'worker'}:
        raise ValueError('installed_gate_roles')
    for process in value['processes']:
        closed(process, 'id image started pid start_ticks pid_namespace cgroup')
        owner = owners.get(process['id'])
        if owner is None or process['id'] in seen or any(process[k] != owner[k] for k in ('id','image','started')):
            raise ValueError('installed_gate_process_binding')
        if any(type(process[k]) is not int or not 1 <= process[k] < 2**64 for k in ('pid','start_ticks','pid_namespace')):
            raise ValueError('installed_gate_process_identity')
        if (process['pid_namespace'] in namespaces or process['cgroup'] not in
                ('/docker/'+process['id'], '/system.slice/docker-'+process['id']+'.scope')):
            raise ValueError('installed_gate_process_identity')
        namespaces.add(process['pid_namespace']); seen.add(process['id'])
    return value


def decode_projection(raw, *, family=None):
    from service.recovery import host_json
    if type(raw) is not bytes or len(raw) > GATE_LIMIT:
        raise ValueError('installed_gate_budget')
    value = projection(host_json(raw), family=family)
    if raw != encode(value).encode():
        raise ValueError('installed_gate_encoding')
    return value


def _control_identity():
    if (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (10001,)*4:
        raise ValueError('installed_control_identity')
    if os.getgroups() not in ([], [10001]):
        raise ValueError('installed_control_groups')


def _directory(path):
    # The fixed parent is the exact bounded /tmp tmpfs checked by the host.
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        if (not stat.S_ISDIR(s.st_mode) or (s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode)) != (10001,10001,0o700) or
                ip.identity(os.stat(path, follow_symlinks=False)) != ip.identity(s)):
            raise ValueError('installed_gate_directory')
        return fd
    except BaseException:
        os.close(fd); raise


def _directory_stable(fd, path):
    # Directory timestamps legitimately change on exclusive publication.
    a = os.fstat(fd); b = os.stat(path, follow_symlinks=False)
    if (a.st_uid,a.st_gid,stat.S_IMODE(a.st_mode)) != (10001,10001,0o700):
        raise ValueError('installed_gate_directory_changed')
    fields = ('st_dev','st_ino','st_mode','st_uid','st_gid')
    if any(getattr(a,k) != getattr(b,k) for k in fields):
        raise ValueError('installed_gate_directory_changed')


def publish(raw, *, directory=GATE_DIRECTORY, family=None):
    """Fixed exec helper only: bounded stdin, exclusive file, no path/action input."""
    _control_identity(); decode_projection(raw, family=family)
    fd = _directory(directory)
    try:
        out = os.open(GATE_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        with os.fdopen(out, 'wb') as stream:
            s = os.fstat(stream.fileno())
            if (s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode),s.st_nlink) != (10001,10001,0o600,1):
                raise ValueError('installed_gate_file')
            if stream.write(raw) != len(raw): raise OSError('installed_gate_short_write')
            stream.flush(); os.fsync(stream.fileno())
            identity = ip.identity(os.fstat(stream.fileno()))
        os.fsync(fd)
        if ip.identity(os.stat(GATE_NAME,dir_fd=fd,follow_symlinks=False)) != identity:
            raise ValueError('installed_gate_replaced')
        _directory_stable(fd,directory)
    finally:
        os.close(fd)


class WaitingControl:
    """One incarnation, one notification; no stale-directory reuse or repair."""
    def __init__(self, *, directory=GATE_DIRECTORY, seconds=30, family=None):
        if family is not None:
            from service import installed_pocket_daemon as pd
            pd.require(family)
        self.family = family
        _control_identity()
        if type(seconds) not in (int,float) or not 0 < seconds <= 30:
            raise ValueError('installed_wait_budget')
        self.directory = Path(directory); self.deadline = time.monotonic()+seconds
        self.directory.mkdir(mode=0o700)  # Existing gate/directory always refuses.
        self.fd = _directory(self.directory)
        self.saved = None

    def read(self):
        _directory_stable(self.fd,self.directory)
        fd = os.open(GATE_NAME,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,dir_fd=self.fd)
        try:
            s = os.fstat(fd)
            if (not stat.S_ISREG(s.st_mode) or (s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode),s.st_nlink) != (10001,10001,0o600,1) or
                    not 1 <= s.st_size <= GATE_LIMIT):
                raise ValueError('installed_gate_file')
            raw = os.read(fd,GATE_LIMIT+1)
            identity = ip.identity(s)
            if (ip.identity(os.fstat(fd)) != identity or
                    ip.identity(os.stat(GATE_NAME,dir_fd=self.fd,follow_symlinks=False)) != identity):
                raise ValueError('installed_gate_replaced')
            value = decode_projection(raw, family=self.family)
            pair = raw,identity
            if self.saved is not None and self.saved != pair:
                raise ValueError('installed_gate_replaced')
            self.saved = pair
            _directory_stable(self.fd,self.directory)
            return value
        finally:
            os.close(fd)

    def wait(self):
        while time.monotonic() < self.deadline:
            try: return self.read()
            except FileNotFoundError:
                # A missing directory is replacement, not an absent notification.
                _directory_stable(self.fd,self.directory)
                time.sleep(.02)
        raise ValueError('installed_activation_timeout')

    def close(self):
        os.close(self.fd)  # Retain notification; never erase failed evidence.


def local_incarnation():
    """Current PID1 in the private PID/cgroup namespaces; no host /proc access."""
    if sys.platform != 'linux' or os.getpid() != 1:
        raise ValueError('installed_control_pid1_required')
    return _read_pid1_incarnation()


def _read_pid1_incarnation():
    if sys.platform != 'linux': raise ValueError('installed_linux_required')
    def read(name):
        with open('/proc/1/'+name,'rb') as stream: raw = stream.read(65537)
        if len(raw)>65536: raise ValueError('installed_proc_budget')
        return raw.decode('ascii')
    def capture():
        raw = read('stat'); fields = raw[raw.rfind(')')+2:].split()
        return dict(start_ticks=int(fields[19]),pid_namespace=os.stat('/proc/1/ns/pid').st_ino,
                    cgroup=read('cgroup').strip())
    result = capture()
    if result != capture() or result['cgroup'] != '0::/':
        raise ValueError('installed_control_incarnation')
    return result


def checked_waiting_gate(waiting, configuration, runtime, *, root='/control/core', launch_policy=None, family=None):
    """Pre-owner check only. The caller must repeat binding after Core acquisition.

    Holding the existing control lock excludes registrars/owners through all
    private copies and live brackets. This function never constructs an owner.
    """
    from service.recovery import installed_startup_read
    _control_identity()
    gate = waiting.wait()
    with ip.locked_file(Path(root)/'control.lock',policy=ip.family_policy(family, configuration),role='control'):
        before = local_incarnation()
        hello = runtime.call('hello', {})
        checked = installed_startup_read(root,configuration,gate,hello,before,policy=ip.family_policy(family, configuration),launch_policy=launch_policy,family=family)
        if (runtime.call('hello', {}) != hello or local_incarnation() != before or
                waiting.read() != gate):
            raise ValueError('installed_startup_bracket_changed')
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('notify',))
    parser.parse_args()
    try:
        publish(sys.stdin.buffer.read(GATE_LIMIT+1))
    except (OSError,ValueError,TypeError,KeyError):
        # No raw request, credential, path, or protected provenance in logs.
        raise SystemExit('installed notification refused') from None



class InstalledActivation:
    """Internal startup dependency; public projection alone never activates."""
    def __init__(self, waiting, configuration, *, launch_policy=None, family=None):
        from service import installed_export as ex, installed_pocket_daemon as pd
        self.family = family
        if family is not None and launch_policy != pd.launch(family):
            raise ValueError("installed_explicit_launch_policy")
        self.launch_policy = launch_policy
        self.waiting = waiting
        self.configuration = ip.family_configuration(configuration, family)
        self.checked = None
        self.prepared = None
        self.started = False

    def _configuration_matches(self, dispatcher):
        from service import installed_export as ex
        profiles = ex.admission_profiles(self.configuration) if self.family is ex.FAMILY else self.configuration['profiles']
        if (encode(dispatcher.current_profiles) != encode(profiles) or
                encode(dispatcher.budget) != encode(self.configuration['budget']) or
                dispatcher.production_contexts is None or dispatcher.persistent_registration is not None):
            raise ValueError('installed_dispatch_configuration')

    def prepare(self, runtime, root):
        if self.prepared is not None: raise ValueError('installed_activation_reuse')
        self.checked = checked_waiting_gate(self.waiting, self.configuration, runtime, root=root, launch_policy=self.launch_policy, family=self.family)
        self.prepared = runtime, Path(root).absolute()
        self.expectation = None
        from service import installed_export as ex
        if self.family is ex.FAMILY:
            from service.installed_export import expectation
            self.expectation = expectation(self.checked["public"], self.family)

    def precheck(self, dispatcher):
        if self.started: raise ValueError('installed_activation_reuse')
        self.started = True
        self._configuration_matches(dispatcher)
        if dispatcher.audio_exports is not None: raise ValueError("installed_caller_export_mapping")
        if self.prepared is None: self.prepare(dispatcher.runtime, dispatcher.root)
        if self.prepared != (dispatcher.runtime, dispatcher.root.absolute()):
            raise ValueError('installed_activation_owner_binding')

    def check_hello(self, hello):
        gate = self.waiting.read()
        from service import installed_export as ex
        if self.family is ex.FAMILY:
            from service.host_install import _registration_hello
            _registration_hello(hello, family=self.family, configuration=self.configuration)
        if (set(hello) != ({'lane', 'boot', 'quarantined', 'launches', 'export_seal'} if self.family is ex.FAMILY else {'lane', 'boot', 'quarantined', 'launches'}) or
                hello['quarantined'] is not False or type(hello['launches']) is not int or
                not 0 <= hello['launches'] < 64 or
                any(hello[k] != gate['registration'][k] for k in ('lane', 'boot'))):
            raise ValueError('installed_unhealthy_hello')

    def postcheck(self, dispatcher):
        # Core holds its lifetime exclusion now. Reuse only the private reader;
        # never reacquire its flock and never open protected receipts.
        from service.recovery import installed_startup_read
        _control_identity()
        self._configuration_matches(dispatcher)
        gate = self.waiting.read()
        before = local_incarnation()
        hello = dispatcher.runtime.call('hello', {})
        checked = installed_startup_read(dispatcher.root, self.configuration, gate,
                                        hello, before, policy=ip.family_policy(self.family, self.configuration), launch_policy=self.launch_policy, family=self.family, owned_db=dispatcher.core.db)
        if (checked != self.checked or self.waiting.read() != gate or
                dispatcher.runtime.call('hello', {}) != hello or local_incarnation() != before):
            raise ValueError('installed_post_owner_binding_changed')
        from service import installed_export as ex
        if self.family is ex.FAMILY:
            from service.audio_export import AudioExports
            export = self.configuration['audio_export']
            dispatcher.audio_exports = AudioExports(export['converter'], export['profile'], export['principals'])


def create_installed_app(*, configuration_file, credential_file, root='/control/core',
                         staging='/staging', runtime=None, waiting=None, launch_policy=None, family=None):
    """Factory-only ASGI entry, with one owner acquired/closed by lifespan.

    No listener is created here. Explicit files and the trusted host notification
    are required; no environment/default credential or synthetic fallback.
    """
    from service.auth import installed_credentials
    from service.execution_context import installed_configuration, installed_mapping, installed_contexts
    from service.dispatcher import Dispatcher
    from service.supervisor_runtime import SupervisorRuntime
    from service.api import create_voice_app
    from service import installed_export as ex, installed_pocket_daemon as pd
    if family is not None and launch_policy != pd.launch(family):
        raise ValueError("installed_explicit_launch_policy")
    if family is ex.FAMILY:
        from service.auth import private_input
        from service.recovery import host_json
        config = ex.configuration(host_json(private_input(configuration_file, mode=0o440, limit=48000)))
        kokoro = {k:v for k,v in config.items() if k != "audio_export"}
        kokoro.update(version=1, ownership_policy=ip.POLICY)
    else:
        config = installed_configuration(configuration_file, policy=ip.SPLIT)
        config = ip.family_configuration(config, family)
        from service import installed_qwen_daemon as qd
        if ip.qwen(config) and family is not qd.FAMILY: raise ValueError('explicit_qwen_daemon_required')
        kokoro = config
    credentials = installed_credentials(credential_file, config['contexts'])
    if family is ex.FAMILY:
        admitted = {c.principal for c in credentials._credentials if {'read','production-write'} <= c.scopes}
        if not set(config['audio_export']['principals']) <= admitted:
            raise ValueError('installed_export_credential_principals')
    mapping = installed_mapping(kokoro, policy=ip.configuration_policy(kokoro))
    contexts = installed_contexts(kokoro, mapping)
    runtime = runtime or SupervisorRuntime('/ipc', **(dict(family=family, configuration=config) if family is not None else {}))
    def dependencies():
        notification = waiting if waiting is not None else WaitingControl(family=family)
        activation = InstalledActivation(notification, config, launch_policy=launch_policy, family=family)
        return notification, lambda: activation.prepare(runtime, root), lambda: Dispatcher(
            root, staging=staging, runtime=runtime, profiles=ex.admission_profiles(config) if family is ex.FAMILY else config['profiles'],
            budget=config['budget'], production_contexts=contexts, installed_activation=activation)
    return _owned_app(dependencies, lambda owner: create_voice_app(
        owner, credentials=credentials, installed=config))


def _owned_app(dependencies, mount):
    """Shared strong installed ownership, including cancelled acquisition/close."""
    from contextlib import asynccontextmanager
    import asyncio
    from fastapi import FastAPI
    application = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @asynccontextmanager
    async def lifespan(app):
        notification, prepare, acquire = dependencies()
        owner = None
        preparation = acquisition = None
        loop = asyncio.get_running_loop()
        responsibility = dict(notification=notification, owner=None, task=None, error=None,
                              operations={}, cleanup_requested=False, closed=False)
        _INSTALLED_LIFECYCLES[id(responsibility)] = responsibility
        app.state.installed_lifecycle = responsibility

        def advance():
            if not responsibility['cleanup_requested'] or responsibility['closed']:
                return
            task = responsibility['task']
            if task is not None and not task.done():
                return
            if task is not None and not task.cancelled():
                return  # Failed close is unknown, never automatically retried.
            if loop.is_closed():
                return
            pending = responsibility.get('waiting_for')
            if pending is not None and not pending.result.done():
                return
            close = responsibility['operations'].get('close')
            if close is not None and close.result.done() and close.result.exception() is not None:
                responsibility['error'] = close.result.exception()
                return
            responsibility['waiting_for'] = pending or acquisition or preparation
            task = loop.create_task(cleanup())
            responsibility['task'] = task
            def finished(done):
                if done.cancelled():
                    advance()
                else:
                    done.exception()
            task.add_done_callback(finished)

        def operation(name, function):
            op = _ThreadResult(loop, function, advance)
            responsibility['operations'][name] = op
            return op

        async def cleanup():
            op = acquisition if acquisition is not None else preparation
            if op is not None:
                responsibility['waiting_for'] = op
                try:
                    result = await op.wait()
                except asyncio.CancelledError as exc:
                    if asyncio.current_task().cancelling() or not op.result.done():
                        raise
                    responsibility['error'] = exc
                except BaseException as exc:
                    responsibility['error'] = exc
                else:
                    if acquisition is not None:
                        responsibility['owner'] = result
                        app.state.dispatcher = result
            acquired = responsibility['owner']
            if acquired is not None:
                close = responsibility['operations'].get('close')
                if close is None:
                    close = operation('close', acquired.close)
                responsibility['waiting_for'] = close
                try:
                    await close.wait()
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    responsibility['error'] = exc
                    raise
            try:
                notification.close()
            except BaseException as exc:
                responsibility['error'] = exc
                raise
            responsibility['closed'] = True
            _INSTALLED_LIFECYCLES.pop(id(responsibility), None)
        try:
            if prepare is not None:
                preparation = operation('preparation', prepare)
                await asyncio.shield(preparation.wrapper)
            acquisition = operation('acquisition', acquire)
            owner = await asyncio.shield(acquisition.wrapper)
            responsibility['owner'] = owner
            app.state.dispatcher = owner
            deadline = time.monotonic()+3
            while True:
                status = await asyncio.to_thread(owner.call, 'status')
                if status['reconciled']: break
                if time.monotonic() >= deadline: raise ValueError('installed_reconciliation_timeout')
                await asyncio.sleep(.02)
            if not status['persistent_activation']: raise ValueError('installed_reconciliation_refused')
            previous = getattr(app.state, 'installed_mount', None)
            if previous is not None:
                app.router.routes.remove(previous)
            app.mount('/', mount(owner))
            app.state.installed_mount = app.router.routes[-1]
            app.state.dispatcher = owner
            yield
        except BaseException as exc:
            # Uvicorn's quiet installed mode suppresses its lifespan traceback.
            # Retain a source location without credential/configuration values.
            trace = exc.__traceback__
            while trace is not None and trace.tb_next is not None:
                trace = trace.tb_next
            if trace is not None:
                import sys
                print('installed_startup_refused:'+type(exc).__name__+':'+
                    trace.tb_frame.f_code.co_name+':'+str(trace.tb_lineno),file=sys.stderr)
            raise
        finally:
            responsibility['cleanup_requested'] = True
            advance()
            await asyncio.shield(responsibility['task'])
    application.router.lifespan_context = lifespan
    return application


STARTUP_NAME = 'startup.json'


def _listener_ready(configuration_digest, *, launch_policy=None):
    """Fixed local authenticated status probe; token/body never leave this helper."""
    from http.client import HTTPConnection
    from service.auth import private_input
    from service.recovery import host_json
    from service import daemon_volumes as dv, installed_export as ex, installed_pocket_daemon as pd
    if launch_policy not in (None, dv.POLICY, ex.LAUNCH, pd.LAUNCH):
        raise ValueError('installed_explicit_launch_policy')
    credential = ip.CREDENTIAL_DESTINATION if launch_policy is None else dv.CREDENTIAL
    value=host_json(private_input(credential,mode=0o640,limit=48000))
    token=value['credentials'][0]['token']
    connection=HTTPConnection('127.0.0.1',8000,timeout=2)
    try:
        connection.request('GET','/skeleton',headers={'Authorization':'Bearer '+token})
        response=connection.getresponse()
        raw=response.read(65537)
        if response.status!=200 or len(raw)>65536: return False
        result=host_json(raw)
        return (result.get('synthetic') is False and (result.get('activation_ready',result.get('ready_for_speech')) is True) and
                result.get('configuration_digest')==configuration_digest and
                result.get('status',{}).get('persistent_activation') is True)
    finally: connection.close()


def launch_status(*, family=None):
    """Fixed exec reader of a nonsecret, incarnation-bound startup result."""
    _control_identity()
    try: fd = _directory(GATE_DIRECTORY)
    except FileNotFoundError:
        print(encode(dict(phase='preparing'))); return
    try:
        try:
            value = os.open(STARTUP_NAME, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK, dir_fd=fd)
        except FileNotFoundError:
            print(encode(dict(phase='waiting'))); return
        try:
            info = os.fstat(value)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                    (info.st_uid,info.st_gid,stat.S_IMODE(info.st_mode)) != (10001,10001,0o600) or info.st_size>4096):
                raise ValueError('installed_startup_file')
            raw = os.read(value,4097)
            if (len(raw)!=info.st_size or ip.identity(os.fstat(value)) != ip.identity(info) or
                    ip.identity(os.stat(STARTUP_NAME,dir_fd=fd,follow_symlinks=False))!=ip.identity(info)):
                raise ValueError('installed_startup_changed')
            result = json.loads(raw)
            from service.recovery import closed
            closed(result,'phase registration_id incarnation')
            if (result['phase']!='ready' or not re.fullmatch('[0-9a-f]{64}',result['registration_id']) or
                    result['incarnation']!=_read_pid1_incarnation()):
                raise ValueError('installed_startup_binding')
            gate=decode_projection(ip.read_private(GATE_DIRECTORY/GATE_NAME,GATE_LIMIT,policy=ip.SPLIT,role='control'), family=family)
            if gate['registration_id']!=result['registration_id']: raise ValueError('installed_startup_gate_binding')
            if not _listener_ready(gate['configuration_digest'],launch_policy=gate.get('launch_policy')):
                print(encode(dict(phase='not_ready',registration_id=result['registration_id']))); return
            print(encode({k:result[k] for k in ('phase','registration_id')}))
        finally: os.close(value)
        _directory_stable(fd,GATE_DIRECTORY)
    finally: os.close(fd)


def _installed_server(config):
    """Normal drain on TERM/INT, with no forced-exit escalation or re-raise.

    Uvicorn's default signal context re-raises TERM after shutdown. The installed
    command instead reports exit 0 only after its retained lifecycle is closed.
    """
    from contextlib import contextmanager
    import signal
    import uvicorn
    class InstalledServer(uvicorn.Server):
        @contextmanager
        def capture_signals(self):
            def stop(*_): self.should_exit = True
            previous={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
            try: yield
            finally:
                for sig,handler in previous.items(): signal.signal(sig,handler)
    return InstalledServer(config)


def installed_main(*, launch_policy=None, family=None):
    """PID1 command factory; Uvicorn listener follows checked owner acquisition."""
    import asyncio
    import uvicorn
    from service import installed_export as ex, installed_pocket_daemon as pd
    if family is not None and launch_policy != pd.launch(family):
        raise ValueError('installed_explicit_launch_policy')
    _control_identity()
    os.umask(0o077)
    if launch_policy is None:
        config_file, credential_file = ip.CONFIG_DESTINATION, ip.CREDENTIAL_DESTINATION
    elif launch_policy == 'installed-daemon-volumes-v1' or family is not None:
        from service import daemon_volumes as dv
        from service.daemon_runtime import publish_control
        publish_control(family=family)
        config_file, credential_file = dv.CONFIG, dv.CREDENTIAL
    else: raise ValueError('installed_explicit_launch_policy')
    app = create_installed_app(configuration_file=config_file, credential_file=credential_file,
        **({} if launch_policy is None else dict(launch_policy=launch_policy, family=family)))
    server = _installed_server(uvicorn.Config(app,host='0.0.0.0',port=8000,access_log=False,
        log_level='critical',timeout_graceful_shutdown=10))

    async def run():
        serving = asyncio.create_task(server.serve())
        try:
            while not server.started:
                if serving.done():
                    await serving
                    raise ValueError('installed_server_not_started')
                await asyncio.sleep(.02)
            lifecycle = app.state.installed_lifecycle
            notification = lifecycle['notification']
            gate = notification.read()
            result = dict(phase='ready',registration_id=gate['registration_id'],incarnation=local_incarnation())
            fd = os.open(STARTUP_NAME,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=notification.fd)
            with os.fdopen(fd,'wb') as stream:
                stream.write(encode(result).encode()); stream.flush(); os.fsync(stream.fileno())
            os.fsync(notification.fd)
            await serving
            if not lifecycle['closed'] or lifecycle['error'] is not None:
                raise ValueError('installed_shutdown_unknown')
        finally:
            # A startup-result failure requests ordinary shutdown, never cancels
            # the owner task or treats an unfinished cleanup as completed.
            server.should_exit = True
            if not serving.done(): await asyncio.shield(serving)
    try:
        asyncio.run(run())
    except BaseException:
        raise SystemExit(2) from None


if __name__ == '__main__':
    main()


def create_native_owner(*, record_file, configuration_file, templates_file, owner_root, staging):
    """Private native owner factory; no listener or readiness promotion.

    The caller owns Dispatcher.close(), including failed/unknown close retention.
    Runtime/contexts are constructed on the existing control-writer thread.
    Retained histories require the separate native completed-history policy.
    """
    from service.dispatcher import Dispatcher
    from service.native_activation import NativeActivation
    activation = NativeActivation(record_file, configuration_file=configuration_file,
        templates_file=templates_file, owner_root=owner_root)
    return Dispatcher(owner_root, staging=staging, installed_activation=activation)


def daemon_main():
    from service.daemon_volumes import POLICY
    installed_main(launch_policy=POLICY)


def daemon_export_main():
    """Explicit trusted new-family PID1 entrypoint."""
    from service.installed_export import FAMILY, LAUNCH
    installed_main(launch_policy=LAUNCH, family=FAMILY)


def publish_export(raw):
    from service.installed_export import FAMILY
    return publish(raw, family=FAMILY)


def export_launch_status():
    from service.installed_export import FAMILY
    return launch_status(family=FAMILY)


def daemon_pocket_main():
    from service.installed_pocket_daemon import FAMILY, LAUNCH
    return installed_main(launch_policy=LAUNCH, family=FAMILY)


def publish_pocket(raw):
    from service.installed_pocket_daemon import FAMILY
    return publish(raw, family=FAMILY)


def pocket_launch_status():
    from service.installed_pocket_daemon import FAMILY
    return launch_status(family=FAMILY)


def daemon_qwen_main():
    from service.installed_qwen_daemon import FAMILY, LAUNCH
    return installed_main(launch_policy=LAUNCH, family=FAMILY)



def publish_qwen(raw):
    from service.installed_qwen_daemon import FAMILY
    return publish(raw, family=FAMILY)



def qwen_launch_status():
    from service.installed_qwen_daemon import FAMILY
    return launch_status(family=FAMILY)
