"""Finite host coordination. No engine, recovery writer or scheduler lives here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import zipfile


def command(role, config, policy, scope):
    """Exact policy used by InstalledObserver; no shell, build or implicit pull."""
    from service import install_policy as ip
    ip.host_policy(policy, config, scope)
    tmp = policy['tmpfs'][role]
    args = ['container', 'create', '--pull=never', '--restart=no', '--read-only',
            '--user', '10001:10001' if role == 'control' else '0:10001',
            '--cap-drop=ALL', '--security-opt=no-new-privileges:true',
            '--ipc=private', '--cgroupns=private', '--network', 'bridge' if role == 'control' else 'none',
            '--memory', str(config['budget']['ram_bytes']), '--memory-swap', str(config['budget']['ram_bytes']),
            '--pids-limit', str(config['budget']['pids']), '--cpus', format(policy['cpu'][role]/10**9, '.9f'),
            '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size='+str(tmp['bytes'])+',nr_inodes='+str(tmp['inodes'])+',mode=1777']
    if role == 'worker':
        args += ['--cap-add=SETUID', '--cap-add=SETGID', '--cap-add=KILL']
    else:
        args += ['--publish', '127.0.0.1:'+str(policy['control_port'])+':8000']
    for name in (('control', 'ipc', 'staging') if role == 'control' else ('ipc', 'receipts', 'staging')):
        args += ['--mount', 'type=volume,source='+scope['volumes'][name]['name']+',destination=/'+name+
                 (',readonly' if role == 'control' and name in ('ipc','staging') else '')]
    if role == 'worker' and ip.pocket(config):
        args += ['--mount', 'type=bind,source='+policy['roots']['control']['source']+'/core/assets,destination=/references/assets,readonly']
    destinations = dict(models='/models', config=ip.CONFIG_DESTINATION, credential=ip.CREDENTIAL_DESTINATION)
    for name in (('models','config','credential') if role == 'control' else ('models','config')):
        source = policy['inputs'][name]['source']
        if ',' in source or '\n' in source: raise ValueError('installed_mount_path')
        args += ['--mount', 'type=bind,source='+source+',destination='+destinations[name]+',readonly']
    args += [config['images'][role], 'python', '-I', '-B', '-c',
             "import sys;sys.path.insert(0,'/opt/speech');from "+
             ('service.entrypoint import installed_main;installed_main()' if role == 'control' else
              "worker.runtime_supervisor import main;main()" )]
    if role == 'worker':
        args += ['--installed-policy', config['ownership_policy'], '--configuration-file', ip.CONFIG_DESTINATION]
    return args


class Launcher:
    """Held host authority is required for the entire finite operation.

    Every Docker invocation is bounded by the existing redacting host transport.
    Failure retains every container and document; never rollback uncertain state.
    """
    def __init__(self, config, scope, policy, authority, directory, *, docker=None, clock=time.monotonic, sleep=time.sleep):
        from service.host_install import Docker
        self.config, self.scope, self.policy = config, scope, policy
        self.authority, self.directory = authority, Path(directory)
        self.docker = docker or Docker(scope['context'])
        self.clock, self.sleep = clock, sleep
        self.end = clock()+90

    def call(self, args):
        from service import recovery
        if self.clock() >= self.end: raise ValueError('installed_launch_timeout')
        # The transport's collection window is per finite operation; the outer
        # coordinator has its own absolute budget covering every invocation.
        self.docker.deadline = min(self.end, self.clock()+recovery.HOST_COLLECTION_SECONDS)
        return self.docker._read(args)

    def document(self, name, value=None):
        from service import host_install as hi
        if value is not None: hi._host_write(self.directory/name, value)
        fd=hi._root_authority_directory(self.directory)
        try:
            raw,_=hi._registration_read_at(fd,name)
            value=hi.recovery.host_json(raw)
            if raw!=hi.encode(value).encode(): raise ValueError('installed_launch_document_changed')
            return value
        finally: os.close(fd)

    def owners(self, ids):
        from service.host_install import _installed_container
        owners=[]
        for role in ('control','worker'):
            self.docker.deadline=min(self.end,self.clock()+5)
            c=_installed_container(self.docker,ids[role])
            if c['image'] != self.config['images'][role]: raise ValueError('installed_image_changed')
            owners.append(dict(id=c['id'], image=c['image'], started=c['started'], role=role,
                volumes=['control','ipc','staging'] if role=='control' else ['ipc','receipts','staging']))
        return owners

    def state(self, identity):
        from service import recovery
        if not recovery.HEX.fullmatch(identity): raise ValueError('installed_container_id')
        value=json.loads(self.call(['container','inspect',identity,'--format',
            '{"running":{{json .State.Running}},"restarting":{{json .State.Restarting}},"exit":{{json .State.ExitCode}}}']))
        recovery.closed(value,'running restarting exit')
        if type(value['running']) is not bool or value['restarting'] is not False or type(value['exit']) is not int:
            raise ValueError('installed_container_state')
        return value

    def marker(self, control):
        # Fixed nonsecret bounded output only. Host never reads credential bytes.
        code = "import sys;sys.path.insert(0,'/opt/speech');from service.entrypoint import launch_status;launch_status()"
        return json.loads(self.call(['exec','--user','10001:10001',control,'python','-I','-B','-c',code]))

    def wait_control(self, identity):
        while self.clock()<self.end:
            if not self.state(identity)['running']: raise ValueError('installed_control_exited')
            marker=self.marker(identity)
            if marker=={'phase':'waiting'}: return
            if marker!={'phase':'preparing'}: raise ValueError('installed_waiting_phase_required')
            self.sleep(.05)
        raise ValueError('installed_waiting_timeout')

    def ready(self, ids, rid):
        original=self.owners(ids)
        while self.clock() < self.end:
            if not all(self.state(i)['running'] for i in ids.values()):
                raise ValueError('installed_owner_exited')
            result=self.marker(ids['control'])
            if result.get('phase')=='ready':
                if result != dict(phase='ready',registration_id=rid): raise ValueError('installed_readiness_binding')
                if self.owners(ids)!=original: raise ValueError('installed_readiness_incarnation')
                self.authority.confirm_ready(original,rid)
                return dict(phase='ready', registration_id=rid, containers=ids)
            if result != {'phase':'waiting'}: raise ValueError('installed_readiness_unknown')
            self.sleep(.05)
        raise ValueError('installed_launch_timeout')

    def retained(self):
        from service import recovery
        value=self.document('containers.json')
        recovery.closed(value,'version configuration_digest containers')
        if value['version']!=1 or value['configuration_digest']!=recovery.digest(self.config):
            raise ValueError('installed_container_binding')
        recovery.closed(value['containers'],'control worker')
        if len(set(value['containers'].values()))!=2 or any(not recovery.HEX.fullmatch(i) for i in value['containers'].values()):
            raise ValueError('installed_container_id')
        return value['containers']

    def up(self):
        from service import recovery
        fresh=False
        try: ids=self.retained()
        except FileNotFoundError:
            # Exclusive intent before the first create. Interrupted creation is
            # retained as an obligation, never repeated or guessed from names.
            self.document('creation-intent.json',dict(version=1,configuration_digest=recovery.digest(self.config)))
            self.authority.initialize_authority()
            fresh=True
            ids={}
            for role in ('control','worker'):
                image=self.call(['image','inspect',self.config['images'][role],'--format','{{.Id}}'])
                if image!=self.config['images'][role]: raise ValueError('installed_local_image_required')
            for role in ('control','worker'):
                identity=self.call(command(role,self.config,self.policy,self.scope))
                if not recovery.HEX.fullmatch(identity): raise ValueError('installed_container_id')
                ids[role]=identity
                self.document('created-'+role+'.json',dict(id=identity))
            self.document('containers.json',dict(version=1,configuration_digest=recovery.digest(self.config),containers=ids))
        states=[self.state(ids[r]) for r in ('control','worker')]
        if all(s['running'] for s in states):
            marker=self.marker(ids['control'])
            if marker.get('phase')!='ready': raise ValueError('installed_live_not_ready')
            return self.ready(ids,marker['registration_id'])
        if any(s['running'] or s['exit']!=0 for s in states): raise ValueError('installed_dirty_retained')
        if not fresh: self.authority.preflight(self.owners(ids))
        self.call(['container','start',ids['control']])
        self.wait_control(ids['control'])
        self.call(['container','start',ids['worker']])
        self.authority.wait_worker()
        owners=self.owners(ids)
        rid=self.authority.register_and_notify(owners)
        return self.ready(ids,rid)

    def status(self):
        ids=self.retained()
        states={r:self.state(i) for r,i in ids.items()}
        if all(s['running'] for s in states.values()):
            marker=self.marker(ids['control'])
            if marker.get('phase')=='ready': return self.ready(ids,marker['registration_id'])
        return dict(phase='retained',states=states,containers=ids)

    def wait_exit(self, identity):
        while self.clock()<self.end:
            s=self.state(identity)
            if not s['running']:
                if s['exit']!=0: raise ValueError('installed_shutdown_unknown')
                return
            self.sleep(.05)
        raise ValueError('installed_shutdown_timeout')

    def stop(self):
        ids=self.retained()
        current=self.status()
        if current['phase']!='ready': raise ValueError('installed_stop_requires_ready')
        # TERM only. Never docker stop (which escalates to KILL), reset or rm.
        self.call(['container','kill','--signal=TERM',ids['control']])
        self.wait_exit(ids['control'])
        self.authority.clean_shutdown(self.owners(ids))
        self.call(['container','kill','--signal=TERM',ids['worker']])
        self.wait_exit(ids['worker'])
        self.authority.preflight(self.owners(ids))
        return dict(phase='stopped',containers=ids)


def archive(source, destination):
    """Deterministic source distribution, explicit allowlist, no repo at install."""
    root=Path(source)
    manifest=json.loads((root/'headless-dist-manifest.json').read_text())
    paths=set(manifest['files'])
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as out:
        launcher=zipfile.ZipInfo('speech-box/speech-box',(1980,1,1,0,0,0))
        launcher.external_attr=0o100755<<16
        out.writestr(launcher,"#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nfrom service.package_install import main\nmain()\n")
        for name in sorted(paths):
            path=root/name
            if (Path(name).is_absolute() or '..' in Path(name).parts or
                    any(part.is_symlink() for part in (path,*path.parents) if part!=root and root in part.parents) or
                    not path.is_file()):
                raise ValueError('package_source_invalid')
            info=zipfile.ZipInfo('speech-box/'+name,(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o100755 if name=='scripts/headless-install.py' else 0o100644)<<16
            out.writestr(info,path.read_bytes())
    return hashlib.sha256(Path(destination).read_bytes()).hexdigest()


def main(argv=None):
    arguments = argv if argv is not None else sys.argv[1:]
    if arguments[:1] == ['docker-models']:
        from service.shared_model_client import main as models_main
        return models_main(arguments[1:])
    if arguments[:1] == ['qwen-snapshot']:
        from service.qwen_snapshot import main as snapshot_main
        return snapshot_main(arguments[1:])
    if arguments[:1] == ['qwen-local-init']:
        from service.qwen_local_config import main as local_main
        return local_main(arguments[1:])
    if arguments[:1] == ['qwen-service-config']:
        from service.qwen_service_config import main as service_main
        return service_main(arguments[1:])
    if arguments[:1] == ['qwen-service-setup']:
        from service.qwen_service_setup import main as setup_main
        return setup_main(arguments[1:])
    if arguments[:1] in (['docker'], ['docker-export'], ['docker-pocket'], ['docker-qwen']):
        from service.daemon_client import main, export_main, pocket_main, qwen_main
        if arguments[0] == 'docker-qwen': return qwen_main(arguments[1:])
        if arguments[0] == 'docker-pocket': docker_main = pocket_main
        elif arguments[0] == 'docker-export': docker_main = export_main
        else: docker_main = main
        return docker_main(arguments[1:])
    parser=argparse.ArgumentParser(description='Finite Linux Docker speech-box installer; retained failures require operator review.',
        epilog='Docker model store: docker-models (init/import/verify/recover-import; use COMMAND --help). Docker volume commands: docker, docker-export, docker-pocket, docker-qwen (install/up/status/stop; use COMMAND --help). Two-model Qwen setup: qwen-service-setup --help.')
    parser.add_argument('command',choices=('init','complete','up','stop','status','archive','setup'))
    parser.add_argument('--configuration'); parser.add_argument('--scope'); parser.add_argument('--host-policy')
    parser.add_argument('--request'); parser.add_argument('--setup-bundle')
    parser.add_argument('--initialization-digest'); parser.add_argument('--output')
    args=parser.parse_args(argv)
    try:
        if args.command=='setup':
            from service.cpu_setup import main as setup
            if not args.request or not args.output: raise ValueError('explicit_setup_request_output_required')
            return setup(['--request',args.request,'--output',args.output])
        if args.command=='archive':
            if not args.output: raise ValueError('output_required')
            print(archive(Path(__file__).resolve().parents[1],args.output)); return
        if sys.version_info[:2] != (3,11): raise ValueError('trusted_python311_required')
        from service import host_install as hi, install_policy as ip
        from service.persistent_init import initialize_split, digest
        if not args.setup_bundle and (not args.configuration or not args.scope): raise ValueError('explicit_inputs_required')
        # Public data only; never accept credential data as an argument.
        def read(path):
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            with os.fdopen(fd,'rb') as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode): raise ValueError('input_type')
                raw=stream.read(65537)
            if len(raw)>65536: raise ValueError('input_budget')
            return hi.recovery.host_json(raw)
        if args.setup_bundle:
            if args.configuration or args.scope or args.host_policy: raise ValueError('setup_bundle_exclusive')
            bundle=read(args.setup_bundle)
            hi.recovery.closed(bundle,'phase layout configuration scope host_policy')
            if bundle['phase']!='configuration-only': raise ValueError('setup_configuration_only_required')
            config=ip.configuration(bundle['configuration']); scope=bundle['scope']
        else:
            config=ip.configuration(read(args.configuration)); scope=read(args.scope)
        hi.validate_scope(scope)
        if args.command!='complete' and args.initialization_digest:
            raise ValueError('completion_command_required')
        os.umask(0o077)
        if args.command=='init':
            if not args.host_policy and not args.setup_bundle: raise ValueError('explicit_host_policy_required')
            hi.create_installed_authority(config,scope,bundle['host_policy'] if args.setup_bundle else read(args.host_policy))
        elif args.host_policy: raise ValueError('provisioning_only_at_init')
        directory=Path(hi.recovery.HOST_AUTHORITY_DIRECTORY)/'installed'/digest(scope)
        with hi.installed_launch_authority(config,scope) as authority:
            policy=authority.policy
            if args.command in ('init','complete'):
                if args.command=='complete' and not re.fullmatch('[0-9a-f]{64}',args.initialization_digest or ''):
                    raise ValueError('initialization_digest_required')
                result=initialize_split({r:v['source'] for r,v in policy['roots'].items()},scope,config,
                    authority=authority.initialize_authority,policy=ip.configuration_policy(config),complete=args.initialization_digest)
                result=dict(phase='initialized',result=result)
            else:
                result=getattr(Launcher(config,scope,policy,authority,directory),args.command)()
        # Context exit performs mandatory final authority/policy stability checks.
        # Publish only after every held-context check and cleanup has succeeded.
        print(json.dumps(result,sort_keys=True))
    except (Exception, KeyboardInterrupt):
        # Arbitrary backend errors can contain paths or secrets. Keep output fixed.
        raise SystemExit('speech-box refused; retained state requires review') from None
