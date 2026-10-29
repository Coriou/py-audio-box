"""Build the closed two-model installed Qwen configuration from trusted templates.

This is a source/installation input builder. It does not fetch weights, write a
Docker volume, launch a daemon, or claim runtime qualification.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import stat

from service import install_policy as ip
from service import installed_qwen_daemon as qd
from worker.runtime_protocol import execution, resident_qwen_model


def build(principal, clone, ryan, profile, budget, images):
    if type(principal) is not str:
        raise ValueError('qwen_service_principal')
    contexts = {'clone': copy.deepcopy(clone), 'Ryan': copy.deepcopy(ryan)}
    specs = {name: execution(context) for name, context in contexts.items()}
    if any(resident_qwen_model(spec) != name for name, spec in specs.items()):
        raise ValueError('qwen_service_exact_models_required')
    from contracts import validate as oracle
    if (clone['plan']['resolved']['profile'] != oracle.profile_snapshot(profile) or
            ryan['plan']['resolved']['profile'] != oracle.profile_snapshot(profile)):
        raise ValueError('qwen_service_shared_profile_required')
    bundles = {}
    for name, spec in specs.items():
        if spec.bundle_id != spec.manifest_sha256:
            contexts[name]['execution']['descriptor']['bundle_id']=spec.manifest_sha256
            contexts[name]['execution']['bundles']={spec.manifest_sha256:'/models/bundles/'+spec.manifest_sha256}
            spec=execution(contexts[name])
        if spec.bundle_id in bundles:
            raise ValueError('qwen_service_distinct_bundles_required')
        bundles[spec.bundle_id] = dict(manifest_sha256=spec.manifest_sha256,
            checkpoint=spec.checkpoint,revision=spec.revision,
            path='/models/bundles/'+spec.bundle_id)
    result = dict(version=1,ownership_policy=qd.POLICY,profiles=[copy.deepcopy(profile)],
        budget=copy.deepcopy(budget),contexts={principal:contexts},bundles=bundles,
        images=copy.deepcopy(images),mounts=dict(control='/control',staging='/staging',
            receipts='/receipts',ipc='/ipc',models='/models'))
    return ip.configuration(result)


def _read(path):
    from service.daemon_client import parse, file_identity
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 65536:
            raise ValueError('qwen_service_configuration_budget')
        raw=stream.read(65537)
        if (len(raw) != before.st_size or file_identity(os.fstat(stream.fileno())) != file_identity(before) or
                file_identity(os.stat(path,follow_symlinks=False)) != file_identity(before)):
            raise ValueError('qwen_service_configuration_changed')
        return parse(raw)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Build and validate the two-model installed Qwen configuration; no installation or inference.')
    parser.add_argument('--request',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    request=_read(args.request)
    if type(request) is not dict or set(request) != {'principal','clone','Ryan','profile','budget','images'}:
        raise ValueError('qwen_service_closed_request')
    result=build(request['principal'],request['clone'],request['Ryan'],request['profile'],request['budget'],request['images'])
    raw=(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n').encode()
    fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    print(Path(args.output))


if __name__ == '__main__':
    main()
