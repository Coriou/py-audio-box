"""Private native-kokoro-cpu-v1 family. Qwen and Linux codecs remain closed.

Only transport framing/scalar/envelope primitives are shared. Selection comes
from trusted factory/frozen profile, never a received receipt/version switch.
"""
from datetime import datetime
import hashlib
from worker import native_protocol as common
from worker.engines.native_kokoro import relation, PROFILE
from worker.engines.qwen_execution import inventory_wire, inventory_from_wire
from worker.hardware import DeviceFamily
from worker.transport import Refused, canonical

VERSION = 'native-kokoro-cpu-v1'
RECEIPT_VERSION = 'native-kokoro-cpu-receipt-v1'
JOURNAL_VERSION = 'native-kokoro-cpu-journal-v1'
OPS = common.OPS - {'put_input'}
FRAME_LIMIT = common.FRAME_LIMIT
CHUNK_LIMIT = common.CHUNK_LIMIT
_closed, _hex, _int = common._closed, common._hex, common._int
runtime_id, chunk = common.runtime_id, common.chunk
send, receive = common.send, common.receive
numeric_bind, tls_context = common.numeric_bind, common.tls_context


def inventory(value):
    observed=inventory_from_wire(value)
    if (observed.host.system,observed.host.architecture,observed.host.deployment)!=('darwin','arm64','native'):
        raise Refused('native_cpu_host')
    if len(observed.devices)!=1: raise Refused('native_cpu_inventory')
    device=observed.devices[0]
    if (device.family,device.index,device.supported)!=(DeviceFamily.CPU,0,True):
        raise Refused('native_cpu_inventory')
    return observed


def request(value):
    return common._request(value,VERSION,OPS)


def receipt(value, *, expected):
    from worker.native_cpu_resources import validate as resources
    keys=('version','runtime_id','binding','deployment_id','configuration_sha256','lane','boot','reaped','timely','original_deadline','reap_method','processes','receiver_closed')
    _closed(value,(*keys,'resources','error','termination'))
    common._receipt({k:value[k] for k in keys},expected=expected,version=RECEIPT_VERSION)
    if any(process['start_identity'] is None or not process['start_identity'].isdigit() for process in value['processes']):
        raise Refused('native_cpu_process_identity')
    resources(value['resources'],processes=value['processes'],timely=value['timely'])
    termination=value['termination'];_closed(termination,('reason','signals','signal_error','observed_exit_at'))
    if termination['reason'] not in (None,'cancelled','startup_deadline','request_deadline','native_cpu_identity_changed','native_cpu_rss_lost','native_cpu_memory_limit'):
        raise Refused('native_cpu_termination')
    signals=termination['signals']
    if type(signals) is not list or len(signals)>2:raise Refused('native_cpu_termination')
    import math
    for entry in signals:
        _closed(entry,('signal','requested_at','accepted','error'))
        if entry['signal'] not in ('SIGTERM','SIGKILL') or type(entry['accepted']) is not bool or type(entry['requested_at']) not in (int,float) or not math.isfinite(entry['requested_at']):raise Refused('native_cpu_termination')
        if entry['error'] is not None:
            _closed(entry['error'],('type','errno'))
            if type(entry['error']['type']) is not str or len(entry['error']['type'])>128 or entry['error']['errno'] is not None and type(entry['error']['errno']) is not int:raise Refused('native_cpu_termination')
    failure=termination['signal_error']
    if failure is not None:
        _closed(failure,('type','message','errno'))
        if any(type(failure[k]) is not str or len(failure[k])>1000 for k in ('type','message')) or failure['errno'] is not None and type(failure['errno']) is not int:raise Refused('native_cpu_termination')
    observed=termination['observed_exit_at']
    if observed is not None and (type(observed) not in (int,float) or not math.isfinite(observed)):raise Refused('native_cpu_termination')
    if value['processes'] and observed is None or value['timely'] and (termination['reason'] is not None or signals or failure is not None):raise Refused('native_cpu_termination')
    error=value['error']
    if error is not None:
        _closed(error,('code','type','message','cause_type','cause'))
        for key in ('code','type','message'):
            if type(error[key]) is not str or len(error[key])>1000:raise Refused('native_cpu_error')
        for key in ('cause_type','cause'):
            if error[key] is not None and (type(error[key]) is not str or len(error[key])>1000):raise Refused('native_cpu_error')
        if value['timely']:raise Refused('native_cpu_error')
    return value


def response(value, *, request):
    result=common._response(value,request=request,inventory_validator=inventory,receipt_validator=receipt)
    if result['ok'] and request['op']=='collect' and result['value']['proposal'] is not None:
        if result['value']['proposal']['manifest']['operation']!='speech_generate':
            raise Refused('native_cpu_preset_only')
    return result


def native_execution(context):
    return relation(context)


def frozen_native(context):
    return context['plan']['resolved']['profile']['id']==PROFILE


def native_binding(context):
    relation(context)
    value=context['execution']['native']
    _closed(value,('deployment_id','configuration_sha256','lane','boot','runtime_manifest_sha256'))
    for key in ('deployment_id','lane','boot'): _hex(value[key],32)
    for key in ('configuration_sha256','runtime_manifest_sha256'): _hex(value[key],64)
    return value


def generated_source(context):
    return 'native-kokoro-execution-sha256:'+hashlib.sha256(canonical(relation(context).identity())).hexdigest()


def terminal_receipt(value, *, authority, runtime_id, local_import_uid=None):
    binding=native_binding(authority.context)
    expected={k:binding[k] for k in ('deployment_id','configuration_sha256','lane','boot')}
    expected.update(runtime_id=runtime_id,binding=authority.binding,original_deadline=authority.job['deadline'])
    receipt(value,expected=expected)
    if local_import_uid is not None: _int(local_import_uid,0,2**31-1)
    return local_import_uid


def owned_preset_context(template,principal,preset,voice_id):
    """Derive catalogue identity only, from an already-enrolled CPU template."""
    import copy
    from service.execution_context import kokoro_preset_version
    from worker.engines.native_kokoro import NativeKokoroExecution
    from lib.spoken_text import freeze
    spec=relation(template);version=kokoro_preset_version(template,principal,preset,voice_id)
    version['reference_asset_sha256']={}
    context=copy.deepcopy(template)
    context['plan']['voice_versions']=[version]
    context['plan']['request']['voice_version_id']=version['id']
    context['execution']['descriptor']=NativeKokoroExecution(spec.model,voice_id,version['id']).identity()
    freeze(context['plan']);relation(context)
    return context
