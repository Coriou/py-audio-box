"""Private, bounded same-incarnation Metal availability; no model acquisition.

One observation and at most one provider thread per supervisor. A stuck provider
cannot spawn replacements. Nothing here is persisted or used as qualification.
"""
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import threading
import time

from worker.engines.qwen_execution import inventory_wire, inventory_from_wire
from worker.hardware import DeviceFamily, HardwareInventory, Device, host_metadata
from worker.transport import Refused, canonical


def digest(inventory):
    return hashlib.sha256(canonical(inventory_wire(inventory))).hexdigest()


def shape(inventory):
    value = inventory_wire(inventory)
    for device in value['devices']:
        device.pop('available'); device.pop('probed_at')
    return canonical(value)


def validate(value, enrolled, manifest, *, now):
    from worker import native_protocol as p
    p._closed(value, ('enrollment_sha256', 'runtime_manifest_sha256', 'inventory'))
    if (value['enrollment_sha256'] != digest(enrolled) or
            value['runtime_manifest_sha256'] != manifest):
        raise Refused('native_availability_binding')
    observed = inventory_from_wire(value['inventory'])
    if shape(observed) != shape(enrolled): raise Refused('native_availability_shape')
    try:
        device = next(d for d in observed.devices if d.family == DeviceFamily.MPS and d.index == 0)
        age = (now - datetime.fromisoformat(device.probed_at)).total_seconds()
        if device.supported is not True or device.available is not True or not 0 <= age <= 300:
            raise ValueError()
    except (StopIteration, TypeError, ValueError):
        raise Refused('stale_or_unavailable_device') from None
    return observed


def host_observation(enrolled):
    """Real host query through Metal's device API, without torch/ML or subprocess.

    Obtaining the default MTLDevice is a cheap availability query, not a model,
    inference, allocation benchmark or qualification. Only exact native arm64
    topology is supported. This function is never called by discovery.
    """
    host = host_metadata('native')
    if host != enrolled.host or host.system != 'darwin' or host.architecture not in ('arm64', 'aarch64'):
        raise Refused('native_availability_shape')
    metal = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
    metal.MTLCreateSystemDefaultDevice.argtypes = []
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    device = metal.MTLCreateSystemDefaultDevice()
    available = bool(device)
    if device:
        objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        objc.objc_msgSend.restype = None
        objc.objc_msgSend(device, objc.sel_registerName(b'release'))
    stamp = datetime.now(timezone.utc).isoformat()
    return HardwareInventory(host, tuple(Device(d.family, d.index, d.supported,
        available if (d.family, d.index) == (DeviceFamily.MPS, 0) else d.available,
        stamp if (d.family, d.index) == (DeviceFamily.MPS, 0) else d.probed_at)
        for d in enrolled.devices))


class Availability:
    def __init__(self, enrolled, manifest, *, provider=host_observation,
                 utc=lambda: datetime.now(timezone.utc), clock=time.monotonic, timeout=1):
        if type(timeout) not in (int, float) or not 0 < timeout <= 1:
            raise Refused('native_availability_timeout')
        self.enrolled = copy.deepcopy(enrolled)
        self.manifest, self.provider, self.utc, self.clock = manifest, provider, utc, clock
        self.timeout = timeout
        self.lock = threading.Lock()
        self.value = None
        self.pending = None

    def validate(self, value, enrolled, manifest, *, now):
        return validate(value,enrolled,manifest,now=now)

    def cached(self):
        with self.lock:
            if self.value is None: return None
            try: self.validate(self.value, self.enrolled, self.manifest, now=self.utc())
            except Refused: return None
            return copy.deepcopy(self.value)

    def get(self):
        value = self.cached()
        if value is not None: return value
        with self.lock:
            if self.value is not None:
                try:
                    self.validate(self.value, self.enrolled, self.manifest, now=self.utc())
                    return copy.deepcopy(self.value)
                except Refused: pass
            if self.pending is None:
                pending = dict(event=threading.Event(), result=None, deadline=self.clock()+self.timeout)
                self.pending = pending
                def run():
                    try:
                        observed = self.provider(copy.deepcopy(self.enrolled))
                        result = dict(enrollment_sha256=digest(self.enrolled),
                            runtime_manifest_sha256=self.manifest, inventory=inventory_wire(observed))
                        self.validate(result, self.enrolled, self.manifest, now=self.utc())
                    except Exception:
                        result = None
                    with self.lock:
                        # Late results never revive a timed-out association.
                        if self.clock() <= pending['deadline']:
                            pending['result'] = result
                            self.value = result
                        pending['event'].set()
                threading.Thread(target=run, name='native-availability', daemon=True).start()
            pending = self.pending
        if not pending['event'].wait(max(0, pending['deadline']-self.clock())):
            raise Refused('native_availability_timeout')
        with self.lock:
            if self.pending is pending: self.pending = None
            value = copy.deepcopy(pending['result'])
        if value is None: raise Refused('native_availability_failed')
        self.validate(value, self.enrolled, self.manifest, now=self.utc())
        return value
