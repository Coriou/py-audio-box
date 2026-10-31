"""CPU-only native metadata observation; never MPS, inference or qualification."""
from datetime import datetime, timezone
from worker.native_availability import Availability as SharedAvailability, digest, shape
from worker import native_cpu_protocol as p
from worker.hardware import HardwareInventory, Device, host_metadata
from worker.transport import Refused


def validate(value,enrolled,manifest,*,now):
    p._closed(value,('enrollment_sha256','runtime_manifest_sha256','inventory'))
    if value['enrollment_sha256']!=digest(enrolled) or value['runtime_manifest_sha256']!=manifest:
        raise Refused('native_cpu_availability_binding')
    observed=p.inventory(value['inventory'])
    if shape(observed)!=shape(enrolled): raise Refused('native_cpu_availability_shape')
    device=observed.devices[0]
    try:
        age=(now-datetime.fromisoformat(device.probed_at)).total_seconds()
        if device.available is not True or not 0<=age<=300: raise ValueError()
    except (TypeError,ValueError): raise Refused('native_cpu_unavailable') from None
    return observed


def host_observation(enrolled):
    from lib.tts_backends.kokoro_native import validate_host, validate_versions
    validate_host(); validate_versions()
    if host_metadata('native')!=enrolled.host: raise Refused('native_cpu_availability_shape')
    p.inventory(p.inventory_wire(enrolled))
    stamp=datetime.now(timezone.utc).isoformat()
    return HardwareInventory(enrolled.host,tuple(Device(d.family,d.index,d.supported,True,stamp) for d in enrolled.devices))


class Availability(SharedAvailability):
    def __init__(self,enrolled,manifest,*,provider=None,**kwargs):
        p.inventory(p.inventory_wire(enrolled))
        super().__init__(enrolled,manifest,provider=host_observation if provider is None else provider,**kwargs)

    def validate(self,value,enrolled,manifest,*,now):
        return validate(value,enrolled,manifest,now=now)
