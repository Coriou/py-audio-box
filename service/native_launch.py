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


if __name__ == '__main__':
    main()
