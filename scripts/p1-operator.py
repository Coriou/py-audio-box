#!/usr/bin/env python3
"""Host-only fixed read-only Docker projection. Never stops or removes containers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from service.core import encode
from service.persistent_init import read_file, validate_scope
from service import recovery


class Docker:
    def __init__(self, context):
        self.context=context

    def _read(self, arguments):
        # No user argv/shell proxy. Each caller below supplies one fixed operation.
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            process=subprocess.Popen(['docker','--context',self.context,*arguments],stdin=subprocess.DEVNULL,stdout=output,stderr=errors)
            end=time.monotonic()+5
            try:
                while process.poll() is None:
                    if output.tell()>1024*1024 or errors.tell()>65536 or time.monotonic()>end:
                        raise ValueError('bounded_docker_inspection_failed')
                    time.sleep(.01)
                if process.returncode or output.tell()>1024*1024: raise ValueError('docker_inspection_failed')
                output.seek(0); return output.read(1024*1024).decode().strip()
            finally:
                if process.poll() is None: process.kill()
                process.wait(timeout=2)

    def observe(self, scope, registration):
        validate_scope(scope); recovery.validate_registration(registration)
        if self.context!=scope['context']: raise ValueError('docker_context_mismatch')
        daemon=self._read(['info','--format','{{.ID}}'])
        if daemon!=scope['daemon']: raise ValueError('docker_daemon_mismatch')
        for role,v in scope['volumes'].items():
            raw=self._read(['volume','inspect',v['name'],'--format','{{json .Name}} {{json .Driver}} {{json .CreatedAt}}'])
            decoder=json.JSONDecoder(); values=[]
            while raw:
                value,n=decoder.raw_decode(raw); values.append(value); raw=raw[n:].lstrip()
            if values!=[v['name'],v['driver'],v['created']]: raise ValueError('docker_volume_mismatch')
        ids=self._read(['container','ls','--all','--quiet','--no-trunc']).splitlines()
        if len(ids)>128 or any(not recovery.HEX.fullmatch(i) for i in ids): raise ValueError('docker_container_budget')
        known={c['id']:c for c in registration['containers']}; observed=[]; states=[]
        names={v['name']:role for role,v in scope['volumes'].items()}
        for identity in sorted(ids):
            # Select only identity/topology fields; never Config.Env or raw receipt.
            raw=self._read(['container','inspect',identity,'--format',
                '{"id":{{json .Id}},"image":{{json .Image}},"started":{{json .State.StartedAt}},"running":{{json .State.Running}},"restarting":{{json .State.Restarting}},"pid":{{json .State.Pid}},"finished":{{json .State.FinishedAt}},"restart":{{json .HostConfig.RestartPolicy.Name}},"pidmode":{{json .HostConfig.PidMode}},"mounts":[{{range $i,$m := .Mounts}}{{if $i}},{{end}}{"type":{{json $m.Type}},"name":{{json $m.Name}},"destination":{{json $m.Destination}},"rw":{{json $m.RW}}}{{end}}]}'])
            value=json.loads(raw)
            mounts=[m for m in value['mounts'] if m['type']=='volume' and m['name'] in names]
            if not mounts: continue
            if identity not in known: raise ValueError('unknown_volume_consumer')
            if len(mounts)!=len(value['mounts']): raise ValueError('unexpected_container_mount')
            expected=known[identity]
            roles=sorted(names[m['name']] for m in mounts)
            if len(roles)!=len(set(roles)) or value['restart'] not in ('no','') or value['pidmode']!='': raise ValueError('unsupported_container_topology')
            for m in mounts:
                role=names[m['name']]
                if m['destination']!='/'+role or (expected['role']=='control' and role in ('ipc','staging') and m['rw']):
                    raise ValueError('container_mount_mismatch')
            actual=dict(id=identity,image=value['image'],started=value['started'],role=expected['role'],volumes=roles)
            if actual!={**expected,'volumes':sorted(expected['volumes'])}: raise ValueError('container_incarnation_mismatch')
            observed.append(expected)
            states.append({k:value[k] for k in ('id','running','restarting','pid','started','finished')})
        if {c['id'] for c in observed}!=set(known): raise ValueError('missing_registered_container')
        return dict(scope=scope,containers=registration['containers'],states=states,evidence='actual-host-inspection')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','inspect','plan','apply'))
    parser.add_argument('--request',required=True)
    args=parser.parse_args()
    if args.command=='apply': recovery.apply()
    request=json.loads(read_file(Path(args.request)))
    recovery.closed(request,'roots scope registration')
    if args.command=='register':
        from service.supervisor_runtime import SupervisorRuntime
        hello=SupervisorRuntime(request['roots']['ipc']).call('hello',{})
        registration=request['registration']
        if (hello.get('lane')!=registration['lane'] or hello.get('boot')!=registration['boot'] or
            hello.get('quarantined') is not False or registration['evidence']!='actual-host-inspection'):
            raise ValueError('authenticated_supervisor_registration_mismatch')
        observer=Docker(request['scope']['context'])
        first=observer.observe(request['scope'],registration)
        if first!=observer.observe(request['scope'],registration): raise ValueError('host_state_changed')
        print(encode(dict(registration_id=recovery.register(request['roots'],request['scope'],registration),evidence='actual-host-inspection')))
        return
    observation=Docker(request['scope']['context']).observe(request['scope'],request['registration'])
    if args.command=='plan': result=recovery.plan(request['roots'],request['scope'],observation)
    else:
        state=recovery.inspect(request['roots'],request['scope']); recovery.check_consumers(state,observation)
        result={k:v for k,v in state.items() if k!='registrations'}
        result['evidence']='actual-host-inspection'
    print(encode(result))


if __name__=='__main__': main()
