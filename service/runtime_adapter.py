"""Trusted host *test harness* Docker operator; no deployed control capability.

Immutable daemon/container/image identities and Docker's exited state supply
container termination evidence. This qualifies only the fixed synthetic process,
not arbitrary descendants or a deployed control capability.
"""
import json
import subprocess
from worker.lifecycle import Observation, State


class DockerHarnessRuntime:
    def __init__(self, invoke=None):
        self.invoke = invoke or self._invoke
        self._started = {}

    @staticmethod
    def _invoke(args, timeout):
        return subprocess.run(['docker', *args], check=True, capture_output=True, text=True, timeout=timeout).stdout

    def identity(self, container, timeout=2):
        daemon = self.invoke(['info', '--format', '{{.ID}}'], timeout).strip()
        item = json.loads(self.invoke(['inspect', container], timeout))[0]
        return json.dumps([daemon, item['Id'], item['Image']], separators=(',', ':'))

    def _inspect(self, identity, timeout):
        daemon, cid, image = json.loads(identity)
        if self.invoke(['info', '--format', '{{.ID}}'], timeout).strip() != daemon:
            raise ValueError('daemon_changed')
        item = json.loads(self.invoke(['inspect', cid], timeout))[0]
        if (item['Id'], item['Image']) != (cid, image) or item['RestartCount'] or item['HostConfig']['RestartPolicy']['Name'] not in ('no', ''):
            raise ValueError('runtime_restarted_or_retargeted')
        return item

    def start(self, runtime_id, timeout):
        item = self._inspect(runtime_id, timeout/5)
        if item['State']['Status'] != 'created':
            raise ValueError('only_never_started_runtime')
        self.invoke(['start', item['Id']], timeout/5)
        item = self._inspect(runtime_id, timeout/5)
        self._started[runtime_id] = item['State']['StartedAt']

    def observe(self, runtime_id, timeout):
        try:
            item = self._inspect(runtime_id, timeout/3)
            state = item['State']
            if self._started.get(runtime_id) != state.get('StartedAt') or runtime_id not in self._started:
                raise ValueError('unobserved_start_or_restart')
            if state['Running'] and state['Status'] == 'running':
                value = State.ALIVE
            elif not state['Running'] and state['Status'] == 'exited' and state['Pid'] == 0 and state['FinishedAt'] and not state['Restarting']:
                # Docker wait independently confirms daemon reaping, not a signal ACK.
                code = int(self.invoke(['wait', item['Id']], timeout/3).strip())
                value = State.STOPPED if code == state['ExitCode'] else State.UNKNOWN
            else:
                value = State.UNKNOWN
        except Exception:
            value = State.UNKNOWN
        return Observation(runtime_id, value)

    def _signal_target(self, runtime_id, timeout):
        item = self._inspect(runtime_id, timeout)
        if runtime_id not in self._started or self._started[runtime_id] != item['State']['StartedAt']:
            raise ValueError('unobserved_start_or_restart')
        return item

    def terminate(self, runtime_id, timeout):
        item = self._signal_target(runtime_id, timeout/4)
        self.invoke(['kill', '--signal', 'TERM', item['Id']], timeout/4)
        try:
            self.invoke(['wait', item['Id']], timeout/4)
        except subprocess.TimeoutExpired:
            pass

    def kill(self, runtime_id, timeout):
        item = self._signal_target(runtime_id, timeout/4)
        self.invoke(['kill', '--signal', 'KILL', item['Id']], timeout/4)
        self.invoke(['wait', item['Id']], timeout/4)
