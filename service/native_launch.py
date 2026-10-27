"""Explicit native installed control launch; no enrollment or provisioning."""
import argparse
import asyncio
from pathlib import Path

from service import native_package as package
from service.auth import decode_credentials
from service.entrypoint import _owned_app, create_native_owner, _installed_server


def create_native_app(*, record_file, configuration_file, templates_file, owner_root,
                      staging, credential_file):
    """Own the reviewed factory via the canonical authenticated HTTP lifespan.

    All paths are explicit. Immutable operator inputs and a dedicated account
    remain required; this adds no topology or hardware qualification authority.
    """
    paths = dict(record_file=record_file, configuration_file=configuration_file,
                 templates_file=templates_file, owner_root=owner_root, staging=staging)
    for path in (*paths.values(), credential_file):
        package.literal(str(path))
    package.directory(str(Path(credential_file).parent))
    templates = package.document(templates_file)
    raw = package.read(credential_file, limit=48000)
    principals = {row['principal'] for row in templates['registrations']}
    credentials = decode_credentials(raw, principals)

    def unchanged():
        package.directory(str(Path(credential_file).parent))
        if package.canonical(package.document(templates_file)) != package.canonical(templates) or package.read(credential_file, limit=48000) != raw:
            raise ValueError('native_launch_inputs_changed')

    class Notification:
        def close(self): pass

    def dependencies():
        def acquire():
            unchanged()
            return create_native_owner(**paths)
        return Notification(), None, acquire

    def mount(owner):
        from service.api import create_voice_app
        unchanged()
        return create_voice_app(owner, credentials=credentials, native_installed=True)
    return _owned_app(dependencies, mount)


def main(argv=None):
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in ('renewal-begin','renewal-retire','renewal-provision','renewal-commit'):
        return maintenance_main(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('serve',))
    for name in ('record-file', 'configuration-file', 'templates-file', 'owner-root', 'staging', 'credential-file'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--host', required=True, choices=('127.0.0.1', '::1'))
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535: raise ValueError('native_launch_port')
        import os
        import uvicorn
        os.umask(0o077)
        app = create_native_app(**{name: getattr(args, name) for name in (
            'record_file', 'configuration_file', 'templates_file', 'owner_root', 'staging', 'credential_file')})
        server = _installed_server(uvicorn.Config(app, host=args.host, port=args.port,
            access_log=False, log_level='critical', timeout_graceful_shutdown=10))
        async def run():
            await server.serve()
            lifecycle = getattr(app.state, 'installed_lifecycle', None)
            if not server.started or lifecycle is None or not lifecycle['closed'] or lifecycle['error'] is not None:
                raise ValueError('native_launch_shutdown_unknown')
        asyncio.run(run())
    except BaseException:
        raise SystemExit('native installed launch refused; retain inputs for review') from None


def maintenance_main(argv):
    """Private operator commands only; no HTTP or public provisioning route."""
    from service import native_renewal as renewal
    parser = argparse.ArgumentParser(description='Private stopped native renewal')
    commands = parser.add_subparsers(dest='command',required=True)
    begin = commands.add_parser('renewal-begin')
    for name in ('record-file','configuration-file','templates-file','owner-root','request-id'):
        begin.add_argument('--'+name,required=True)
    retire = commands.add_parser('renewal-retire')
    retire.add_argument('--intent-file',required=True); retire.add_argument('--host-root',required=True)
    provision = commands.add_parser('renewal-provision')
    for name in ('intent-file','export-file','host-root'): provision.add_argument('--'+name,required=True)
    commit = commands.add_parser('renewal-commit')
    commit.add_argument('--intent-file',required=True); commit.add_argument('--export-file',required=True)
    args = vars(parser.parse_args(argv)); command = args.pop('command')
    try:
        if command == 'renewal-begin':
            record = args.pop('record_file'); result = renewal.begin(record,**args)
        elif command == 'renewal-retire':
            intent = args.pop('intent_file'); result = renewal.retire(intent,**args)
        elif command == 'renewal-provision':
            intent = args.pop('intent_file'); result = renewal.provision(intent,**args)
        else:
            intent = args.pop('intent_file'); result = renewal.commit(intent,**args)
        print(package.canonical(result).decode())
    except Exception as error:
        raise SystemExit(str(error)+'; retain all inputs and evidence') from None


if __name__ == '__main__':
    main()
