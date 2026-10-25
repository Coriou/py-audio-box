"""Mocked private RPC clocks/transport only; no processes, sleeps or inference."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from service import installed_export as ex, installed_pocket_daemon as pd
from service.supervisor_runtime import SupervisorRuntime
from worker.runtime_protocol import VERSION, POCKET_START_RPC_SECONDS
from worker.runtime_supervisor import RuntimeSupervisor


def runtime(family):
    # Constructor configuration admission is tested separately; this seam selects
    # the already-admitted executable family without provisioning anything.
    value = SupervisorRuntime.__new__(SupervisorRuntime)
    value.family=family; value.interrupt='none'
    value.ipc=Mock(); value.ipc.__truediv__=Mock(return_value=Mock())
    (value.ipc/'lifecycle.key').read_text.return_value='a'*64
    return value


@pytest.mark.parametrize('family,seconds', [(pd.FAMILY,10),(None,1.5),(ex.FAMILY,1.5)])
def test_start_selects_only_trusted_pocket_response_budget(family,seconds):
    value=runtime(family);value.call=Mock(return_value={'state':'alive'})
    value.start('id')
    value.call.assert_called_once_with('start',{'runtime_id':'id'},seconds)


@pytest.mark.parametrize('family,op',[(None,'start'),(ex.FAMILY,'start'),(pd.FAMILY,'prepare'),(pd.FAMILY,'observe')])
def test_long_budget_refused_for_other_families_and_operations(family,op):
    with pytest.raises(ValueError,match='bounded_communication_required'):
        runtime(family).call(op,{},POCKET_START_RPC_SECONDS)


def test_client_auth_and_send_keep_short_original_cutoff():
    value=runtime(pd.FAMILY)
    connection=Mock();connection.__enter__=Mock(return_value=connection);connection.__exit__=Mock(return_value=False)
    with patch('service.supervisor_runtime.time.monotonic',return_value=100), patch('service.supervisor_runtime.socket.socket',return_value=connection), patch('service.supervisor_runtime.authenticate') as auth, patch('service.supervisor_runtime.send') as send, patch('service.supervisor_runtime.receive',return_value={'ok':True,'value':{}}) as receive:
        value.start('id')
    connection.settimeout.assert_called_once_with(2)
    assert auth.call_args.args[2]==102
    assert send.call_args.args[2]==102
    assert receive.call_args.args[1]==110


@pytest.mark.parametrize('pocket,installed,op,budget',[(True,True,'start',10),(False,True,'start',2),(True,False,'start',2),(False,False,'start',2),(True,True,'observe',2)])
def test_server_response_uses_entry_anchor_after_authenticated_request(pocket,installed,op,budget):
    value=RuntimeSupervisor.__new__(RuntimeSupervisor)
    value._pocket_daemon=pocket;value.installed_pocket=installed;value.secret='a'*64
    value.commands=Mock();value.slots=Mock();connection=Mock();answer=Mock()
    # Authentication/request consumes 1.5 seconds; no deadline renewal follows.
    with patch('worker.runtime_supervisor.time.monotonic',side_effect=[100,101.5]), patch('worker.runtime_supervisor.authenticate') as auth, patch('worker.runtime_supervisor.receive',return_value={'version':VERSION,'op':op,'body':{'runtime_id':'id'}}) as receive, patch('worker.runtime_supervisor.queue.Queue',return_value=answer), patch('worker.runtime_supervisor.send') as send:
        value.client(connection)
    assert auth.call_args.args[2]==102
    assert receive.call_args.args[1]==102
    answer.get.assert_called_once_with(timeout=budget-1.5)
    assert send.call_args.args[2]==100+budget
    connection.close.assert_called_once();value.slots.release.assert_called_once()


def test_malformed_start_body_never_gets_long_response_or_queue_entry():
    value=RuntimeSupervisor.__new__(RuntimeSupervisor)
    value._pocket_daemon=True;value.installed_pocket=True;value.secret='a'*64
    value.commands=Mock();value.slots=Mock()
    with patch('worker.runtime_supervisor.time.monotonic',return_value=100), patch('worker.runtime_supervisor.authenticate'), patch('worker.runtime_supervisor.receive',return_value={'version':VERSION,'op':'start','body':{'runtime_id':'id','extra':True}}), patch('worker.runtime_supervisor.send') as send:
        value.client(Mock())
    value.commands.put_nowait.assert_not_called();send.assert_not_called()
