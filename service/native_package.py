"""Offline native-worker package preflight. This is not installed activation.

Operator-supplied identities are checked, never resolved or downloaded here.
A matching environment does not establish ABI, MPS or speech qualification.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import struct
import sys
import zipfile

from service.install_policy import parent_fd, identity
from worker import native_protocol as p
from worker.transport import Refused, canonical

VARIANT = 'native-qwen-v1'
ROOT = Path(__file__).resolve().parents[1]


def literal(path):
    if type(path) is not str or not Path(path).is_absolute() or '..' in Path(path).parts:
        raise Refused('native_package_path')
    return Path(path)


def read(path, limit=1048576, *, mode=0o600):
    """No-follow all components; fixed owner/mode and stable exclusive file."""
    path = literal(str(path))
    with parent_fd(path) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or
                    before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != mode or before.st_size > limit):
                raise Refused('native_package_file')
            raw = b''
            while len(raw) <= limit:
                chunk = os.read(fd, min(65536, limit + 1 - len(raw)))
                if not chunk: break
                raw += chunk
            if (len(raw) != before.st_size or identity(before) != identity(os.fstat(fd)) or
                    identity(before) != identity(os.stat(name, dir_fd=parent, follow_symlinks=False))):
                raise Refused('native_package_changed')
            return raw
        finally:
            os.close(fd)


def document(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value: raise Refused('native_package_duplicate')
            value[key] = item
        return value
    value = json.loads(read(path), object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(Refused('native_package_json')))
    if canonical(value) != read(path): raise Refused('native_package_canonical')
    return value


def directory(path):
    path = literal(path)
    with parent_fd(path) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise Refused('native_package_directory')
        finally: os.close(fd)
    return path


def file_hash(path, *, executable=False):
    """Stream operator artifacts, refusing mutable or foreign-owned files."""
    path = literal(path)
    with parent_fd(path) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or
                    before.st_nlink != 1 or before.st_mode & 0o022 or
                    executable and not before.st_mode & 0o100):
                raise Refused('native_package_artifact')
            digest = hashlib.sha256()
            while True:
                chunk = os.read(fd, 65536)
                if not chunk: break
                digest.update(chunk)
            if (identity(before) != identity(os.fstat(fd)) or
                    identity(before) != identity(os.stat(name, dir_fd=parent, follow_symlinks=False))):
                raise Refused('native_package_changed')
            return digest.hexdigest()
        finally: os.close(fd)


def python_identity():
    return dict(executable=sys.executable, version=sys.version, build=list(platform.python_build()),
                implementation=platform.python_implementation(), machine=platform.machine(), system=platform.system())


def distributions():
    from packaging.utils import canonicalize_name
    result = {}
    for dist in importlib.metadata.distributions():
        name = canonicalize_name(dist.metadata['Name'])
        if name in result: raise Refused('native_package_distribution_duplicate')
        result[name] = dist.version
    return result


def dependency_closure(expected):
    """Exact installed set plus active Requires-Dist edges, without importing ML.

    This checks metadata only. Binary loader/ABI compatibility remains unrun.
    Extras are deliberately unsupported in this first package portion.
    """
    from packaging.utils import canonicalize_name
    if (type(expected) is not dict or not expected or len(expected) > 512 or
            any(type(k) is not str or canonicalize_name(k) != k or type(v) is not str or
                not v or '+' in v for k, v in expected.items())):
        raise Refused('native_package_distributions')
    if not {'packaging', 'torch', 'transformers', 'qwen-tts'} <= set(expected):
        raise Refused('native_package_missing_dependencies')
    if distributions() != expected: raise Refused('native_package_dependency_mismatch')
    return {canonicalize_name(dist.metadata['Name']):
            active_edges(dist.requires or (), expected) for dist in importlib.metadata.distributions()}


def active_edges(requirements, expected):
    """Canonical active graph; inactive valid markers carry no edge authority."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    edges = set()
    if len(requirements) > 512: raise Refused('native_package_dependency_closure')
    try:
        for raw in requirements:
            if len(raw) > 4096: raise Refused('native_package_dependency_closure')
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({'extra': ''}): continue
            name = canonicalize_name(requirement.name)
            if (requirement.url or requirement.extras or name not in expected or
                    expected[name] not in requirement.specifier):
                raise Refused('native_package_dependency_closure')
            edges.add((name, str(requirement.specifier)))
    except (ValueError, TypeError, KeyError):
        raise Refused('native_package_dependency_closure') from None
    return frozenset(edges)


def zip_preflight(path, *, source=False):
    """Bound constructor allocation before ZipFile reads its central directory.

    Single-disk classic ZIP only; ZIP64 and trailing data are unsupported.
    Operator artifacts remain immutable under the documented trusted-writer premise.
    """
    limit = 64 * 1024 * 1024 if source else 4 * 1024 * 1024 * 1024
    with open(path, 'rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        if size > limit or size < 22: raise Refused('native_package_zip_bounds')
        stream.seek(max(0, size - 65557))
        tail = stream.read(65557)
        offset = tail.rfind(b'PK\x05\x06')
        if offset < 0 or len(tail) - offset < 22: raise Refused('native_package_zip_bounds')
        _, disk, central_disk, disk_count, count, central_size, central_offset, comment = struct.unpack_from('<4s4H2IH', tail, offset)
        end = size - len(tail) + offset
        if (disk or central_disk or disk_count != count or count > (512 if source else 32768) or
                central_size > 4 * 1024 * 1024 or central_offset == 0xffffffff or
                central_offset + central_size != end or offset + 22 + comment != len(tail)):
            raise Refused('native_package_zip_bounds')


def zip_read(archive, info, limit):
    if info.file_size > limit: raise Refused('native_package_zip_member')
    with archive.open(info) as stream:
        raw = stream.read(limit + 1)
    if len(raw) != info.file_size or len(raw) > limit: raise Refused('native_package_zip_member')
    return raw


def wheel_identity(path, names, expected):
    """Validate wheel identity, tags and active edges; never infer ABI success."""
    from email.parser import BytesParser
    from packaging.tags import sys_tags, parse_tag
    from packaging.utils import parse_wheel_filename, canonicalize_name
    from packaging.specifiers import SpecifierSet
    if len(names) != 1: raise Refused('native_package_wheel_identity')
    try:
        name, version, _, tags = parse_wheel_filename(Path(path).name)
        if name != names[0] or str(version) != expected[name] or not tags.intersection(set(sys_tags())):
            raise Refused('native_package_wheel_platform')
        zip_preflight(path)
        with zipfile.ZipFile(path) as wheel:
            metadata = [i for i in wheel.infolist() if i.filename.endswith('.dist-info/METADATA')]
            headers = [i for i in wheel.infolist() if i.filename.endswith('.dist-info/WHEEL')]
            if len(metadata) != 1 or len(headers) != 1 or any(i.file_size > 65536 for i in metadata + headers):
                raise Refused('native_package_wheel_metadata')
            info = BytesParser().parsebytes(zip_read(wheel, metadata[0], 65536))
            header = BytesParser().parsebytes(zip_read(wheel, headers[0], 65536))
            actual_tags = set()
            for tag in header.get_all('Tag', []): actual_tags.update(parse_tag(tag))
            if (canonicalize_name(info['Name']) != name or info['Version'] != expected[name] or
                    tags != actual_tags or info['Requires-Python'] and
                    platform.python_version() not in SpecifierSet(info['Requires-Python'])):
                raise Refused('native_package_wheel_metadata')
            return active_edges(info.get_all('Requires-Dist', []), expected)
    except Refused:
        raise
    except (ValueError, TypeError, KeyError, zipfile.BadZipFile):
        raise Refused('native_package_wheel_metadata') from None


def runtime_manifest(value):
    p._closed(value, ('version', 'variant', 'python', 'distributions', 'artifacts',
                      'source_archive', 'models', 'qualification'))
    if (type(value['version']) is not int or value['version'] != 1 or value['variant'] != VARIANT or
            value['qualification'] != 'unverified'):
        raise Refused('native_package_manifest')
    expected = value['python']
    p._closed(expected, ('executable', 'version', 'build', 'implementation', 'machine', 'system', 'sha256'))
    p._hex(expected['sha256'], 64)
    actual = python_identity()
    if (actual['system'] != 'Darwin' or actual['machine'] != 'arm64' or
            {k: v for k, v in expected.items() if k != 'sha256'} != actual):
        raise Refused('native_package_python_identity')
    if file_hash(expected['executable'], executable=True) != expected['sha256']:
        raise Refused('native_package_python_hash')
    installed_edges = dependency_closure(value['distributions'])
    artifacts = value['artifacts']
    if type(artifacts) is not list or not 1 <= len(artifacts) <= 1024:
        raise Refused('native_package_artifacts')
    covered = set(); paths = set()
    for artifact in artifacts:
        p._closed(artifact, ('path', 'sha256', 'distributions'))
        p._hex(artifact['sha256'], 64)
        names = artifact['distributions']
        if (type(names) is not list or not names or len(set(names)) != len(names) or
                not set(names) <= set(value['distributions']) or covered.intersection(names) or
                artifact['path'] in paths): raise Refused('native_package_artifacts')
        if file_hash(artifact['path']) != artifact['sha256']: raise Refused('native_package_artifact_hash')
        edges = wheel_identity(artifact['path'], names, value['distributions'])
        if edges != installed_edges[names[0]]:
            raise Refused('native_package_metadata_disagreement')
        covered.update(names); paths.add(artifact['path'])
    if covered != set(value['distributions']): raise Refused('native_package_missing_artifacts')
    archive = value['source_archive']
    p._closed(archive, ('path', 'sha256'))
    p._hex(archive['sha256'], 64)
    if file_hash(archive['path']) != archive['sha256']: raise Refused('native_package_source_hash')
    verify_archive(archive['path'])
    models = value['models']
    if type(models) is not list or not 1 <= len(models) <= 64: raise Refused('native_package_models')
    seen = set()
    for model in models:
        p._closed(model, ('bundle_id', 'path', 'manifest_sha256', 'checkpoint', 'revision', 'license', 'access'))
        if (model['bundle_id'] in seen or any(type(model[k]) is not str or not 1 <= len(model[k]) <= 512
                for k in ('bundle_id', 'checkpoint', 'revision', 'license', 'access'))):
            raise Refused('native_package_model_metadata')
        seen.add(model['bundle_id']); p._hex(model['manifest_sha256'], 64)
        literal(model['path'])
        # Full model byte/capability verification occurs against each registration below.
    return hashlib.sha256(canonical(value)).hexdigest()


def members():
    value = json.loads((ROOT / 'native-dist-manifest.json').read_bytes())
    if value['variant'] != VARIANT: raise Refused('native_package_variant')
    files = value['files']
    if len(set(files)) != len(files): raise Refused('native_package_members')
    for name in files:
        if Path(name).is_absolute() or '..' in Path(name).parts: raise Refused('native_package_members')
    return files


LAUNCHER = b"import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nfrom service.native_package import main\nmain()\n"


def archive(destination):
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as out:
        for name in sorted(members()):
            path = ROOT / name
            if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
                raise Refused('native_package_source')
            info = zipfile.ZipInfo('native-worker/' + name, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16; info.compress_type = zipfile.ZIP_DEFLATED
            out.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo('native-worker/native-worker.py', (1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        out.writestr(info, LAUNCHER)
    return hashlib.sha256(Path(destination).read_bytes()).hexdigest()


def verify_archive(path):
    zip_preflight(path, source=True)
    with zipfile.ZipFile(path) as source:
        expected = {'native-worker/' + name for name in members()} | {'native-worker/native-worker.py'}
        if len(source.namelist()) != len(expected) or set(source.namelist()) != expected:
            raise Refused('native_package_source_members')
        for name in members():
            info = source.getinfo('native-worker/' + name)
            local = ROOT / name
            if any(part.is_symlink() for part in (local, *local.parents)):
                raise Refused('native_package_source_changed')
            if info.file_size != local.stat().st_size or zip_read(source, info, local.stat().st_size) != local.read_bytes():
                raise Refused('native_package_source_changed')
        launcher = source.getinfo('native-worker/native-worker.py')
        if launcher.file_size != len(LAUNCHER): raise Refused('native_package_launcher')
        if zip_read(source, launcher, len(LAUNCHER)) != LAUNCHER: raise Refused('native_package_launcher')


def preflight(config_path, manifest_path):
    """Return checked existing supervisor configuration; no socket/process/ML."""
    from worker.engines.qwen_execution import QwenExecution
    from worker.engines.qwen import verify_bundle
    config = document(config_path); manifest = document(manifest_path)
    p._closed(config, ('root','deployment_id','configuration_sha256','peer_pin','runtime_manifest_sha256',
        'registrations','python','child_entry','host','port','proven_host_network','cafile','certfile','keyfile'))
    directory(str(literal(str(config_path)).parent)); directory(str(literal(str(manifest_path)).parent))
    directory(config['root'])
    for key, size in (('deployment_id',32), ('configuration_sha256',64), ('peer_pin',64)):
        p._hex(config[key], size)
    if type(config['proven_host_network']) is not bool: raise Refused('native_package_topology')
    p.numeric_bind(config['host'], config['port'], proven_host_network=config['proven_host_network'])
    for key in ('cafile', 'certfile', 'keyfile'):
        directory(str(literal(config[key]).parent)); read(config[key], 65536)
    digest = runtime_manifest(manifest)
    if (config['runtime_manifest_sha256'] != digest or config['python'] != sys.executable or
            config['child_entry'] != str(ROOT / 'worker/native_child.py')):
        raise Refused('native_package_runtime_binding')
    if type(config['registrations']) is not list or not 1 <= len(config['registrations']) <= 64:
        raise Refused('native_package_registrations')
    models = {m['bundle_id']: m for m in manifest['models']}; used = set()
    for registration in config['registrations']:
        p._closed(registration, ('snapshot','execution','bundles','inventory'))
        spec = QwenExecution.from_wire(registration['execution'])
        if spec != p.native_execution(registration['snapshot'][1]): raise Refused('native_package_registration')
        native = p.native_binding(registration['snapshot'][1])
        if any(native[k] != config[k] for k in ('deployment_id', 'configuration_sha256', 'runtime_manifest_sha256')):
            raise Refused('native_package_registration_binding')
        model = models.get(spec.bundle_id)
        if model is None or registration['bundles'] != {spec.bundle_id: model['path']}:
            raise Refused('native_package_model_binding')
        if (spec.manifest_sha256,spec.checkpoint,spec.revision) != (model['manifest_sha256'],model['checkpoint'],model['revision']):
            raise Refused('native_package_model_binding')
        verify_bundle(model['path'], spec.manifest_sha256, spec.checkpoint, spec.revision,
                      spec.language, owner_uid=os.getuid(), execution=spec)
        used.add(spec.bundle_id)
    if used != set(models): raise Refused('native_package_model_binding')
    return config


def enrolled_runtime(configuration_file, *, connector=None):
    """Private control-side enrollment consumer, never public registration.

    Exact configured current boot is required. Existing NativeRuntime owns its
    historical binding/import checks. This factory grants no installed activation
    or readiness and does not reinterpret Linux enrollment/history.
    """
    from service.native_runtime import NativeRuntime
    config = document(configuration_file)
    directory(str(literal(str(configuration_file)).parent))
    p._closed(config, ('version', 'variant', 'native', 'host', 'port', 'peer_sha256',
                      'cafile', 'certfile', 'keyfile', 'spool', 'staging'))
    if type(config['version']) is not int or config['version'] != 1 or config['variant'] != VARIANT:
        raise Refused('native_package_control_variant')
    for key in ('cafile', 'certfile', 'keyfile'):
        directory(str(literal(config[key]).parent)); read(config[key], 65536)
    context = p.tls_context(server=False, **{key: config[key] for key in ('cafile','certfile','keyfile')})
    runtime = NativeRuntime(host=config['host'], port=config['port'], tls=context,
        peer_sha256=config['peer_sha256'], native=config['native'], spool=config['spool'],
        staging=config['staging'], local_import_uid=os.getuid(), connector=connector)
    hello = runtime.call('hello', {})
    if (any(hello[key] != value for key, value in config['native'].items()) or
            hello['quarantined'] is not False or not runtime.admission_open):
        raise Refused('native_package_control_hello')
    return runtime


def serve_checked(config):
    """Consume the checked private snapshot, avoiding a second config read."""
    from worker.native_supervisor import NativeSupervisor, serve
    from worker.engines.qwen_execution import QwenExecution, inventory_from_wire
    registrations = {}
    for entry in config['registrations']:
        binding = hashlib.sha256(canonical(entry['snapshot'])).hexdigest()
        if binding in registrations: raise Refused('native_package_duplicate_registration')
        registrations[binding] = dict(entry, execution=QwenExecution.from_wire(entry['execution']),
                                      inventory=inventory_from_wire(entry['inventory']))
    context = p.tls_context(server=True, **{key: config[key] for key in ('cafile','certfile','keyfile')})
    supervisor = NativeSupervisor(**{key: config[key] for key in ('root','deployment_id',
        'configuration_sha256','peer_pin','runtime_manifest_sha256','python','child_entry')},
        registrations=registrations)
    serve(supervisor, context, config['host'], config['port'],
          proven_host_network=config['proven_host_network'])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Offline native-qwen-v1 SOURCE package; speech remains unqualified.')
    parser.add_argument('command', choices=('archive', 'check', 'serve'))
    parser.add_argument('--output'); parser.add_argument('--config'); parser.add_argument('--runtime-manifest')
    args = parser.parse_args(argv)
    try:
        if args.command == 'archive':
            if not args.output or args.config or args.runtime_manifest: raise Refused('native_package_arguments')
            print(archive(args.output)); return
        if not args.config or not args.runtime_manifest or args.output: raise Refused('native_package_arguments')
        config = preflight(args.config, args.runtime_manifest)
        if args.command == 'check':
            print(json.dumps(dict(variant=VARIANT, package='prepared', dependencies='metadata-verified',
                abi='unrun', qualification='unqualified', speech='unavailable'), sort_keys=True)); return
        serve_checked(config)
    except Exception:
        raise SystemExit('native package refused; retained inputs require review') from None


if __name__ == '__main__': main()
