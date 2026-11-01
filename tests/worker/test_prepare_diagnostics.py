"""Prepare diagnostics are expendable and never change lifecycle decisions."""
import queue
import threading
from unittest.mock import patch

from worker import runtime_supervisor as module
from worker.runtime_protocol import QWEN_PREPARE_RPC_SECONDS


def test_diagnostic_quota_and_queue_drop_without_sink_access():
    class Pending:
        def __init__(self): self.lines=[]
        def full(self): return False  # A draining sink leaves capacity available.
        def put_nowait(self, line): self.lines.append(line)
    pending=Pending()
    with patch.object(module, '_diagnostic_queue', pending), \
         patch.object(module, '_diagnostic_started', True), \
         patch.object(module, '_diagnostic_events', 0), \
         patch.object(module, '_diagnostic_limit', 2), \
         patch.object(module.os, 'write', side_effect=AssertionError('caller touched sink')):
        for n in range(10): module._prepare_phase('handling', 0, n)
        assert module._diagnostic_events == 2
        assert len(pending.lines) == 2
        assert pending.full() is False
        assert pending.lines[0].startswith(b'lifecycle_prepare_phase:0:handling:')
        assert pending.lines[1].startswith(b'lifecycle_prepare_phase:1:handling:')


def test_consumer_keeps_its_queue_when_global_queue_is_replaced():
    first_read = threading.Event()
    second_read = threading.Event()
    release_write = threading.Event()

    class OwnedQueue(queue.Queue):
        def __init__(self):
            super().__init__(32)
            self.reads = 0

        def get(self, *args, **kwargs):
            self.reads += 1
            if self.reads == 2:
                second_read.set()
            return super().get(*args, **kwargs)

    old_queue = OwnedQueue()
    replacement = queue.Queue(32)
    old_queue.put_nowait(b'first line')

    def blocked_write(fd, line):
        first_read.set()
        assert release_write.wait(1)
        return len(line)

    with patch.object(module.os, 'write', side_effect=blocked_write), \
         patch.object(module, '_diagnostic_queue', old_queue):
        consumer = threading.Thread(target=module._diagnostic_consumer,
                                    args=(old_queue,), daemon=True)
        consumer.start()
        try:
            assert first_read.wait(1)
            module._diagnostic_queue = replacement
            release_write.set()
            assert second_read.wait(1)
            replacement.put_nowait(None)
            assert replacement.get_nowait() is None
            assert consumer.is_alive()
        finally:
            release_write.set()
            old_queue.put_nowait(None)
            consumer.join(1)
        assert not consumer.is_alive()


def test_prepare_client_queue_wait_and_response_markers():
    class Connection:
        def close(self): pass
    lane = module.RuntimeSupervisor.__new__(module.RuntimeSupervisor)
    lane.secret = '00' * 32
    lane.commands = queue.Queue(1)
    lane.slots = type('Slots', (), {'release': lambda self: None})()
    markers = []
    request = {'version': module.request.__globals__['VERSION'], 'op': 'prepare',
               'body': {'fixture': 'qwen'}}
    with patch.object(module, 'authenticate'), patch.object(module, 'receive', return_value=request), \
         patch.object(module, '_prepare_phase', side_effect=lambda name, _, seq=0: markers.append((seq,name))), \
         patch.object(module, 'send') as send:
        lane.client(Connection())
        assert [name for _,name in markers] == ['enqueue_intent','server_prepare_response_wait_expired']
        first_sequence=markers[0][0]
        assert all(seq == first_sequence for seq,_ in markers)
        assert not send.called
        markers.clear()
        lane.commands.get_nowait()
        answer = queue.Queue(1)
        answer.put({'ok': False, 'error': 'lifecycle_refused_or_quarantined'})
        with patch.object(module.queue, 'Queue', return_value=answer):
            lane.client(Connection())
        assert 'response_sent_refused' in [name for _,name in markers]
        assert markers[0][0] != first_sequence
        markers.clear()
        lane.commands.get_nowait()
        answer.put({'ok': True, 'value': {'runtime_id': 'synthetic'}})
        with patch.object(module, 'send', side_effect=BrokenPipeError):
            with patch.object(module.queue, 'Queue', return_value=answer):
                lane.client(Connection())
        assert 'response_send_failed' in [name for _,name in markers]


def test_full_command_queue_reports_failure_without_claiming_enqueued():
    class Connection:
        def close(self): pass
    lane=module.RuntimeSupervisor.__new__(module.RuntimeSupervisor)
    lane.secret='00'*32;lane.commands=queue.Queue(1)
    lane.commands.put_nowait(('occupied',))
    lane.slots=type('Slots', (), {'release': lambda self: None})()
    request={'version': module.request.__globals__['VERSION'], 'op': 'prepare',
             'body': {'fixture': 'qwen'}}
    markers=[]
    with patch.object(module,'authenticate'),patch.object(module,'receive',return_value=request), \
         patch.object(module,'_prepare_phase',side_effect=lambda name,_,seq=0:markers.append((seq,name))):
        lane.client(Connection())
    assert [name for _,name in markers]==['enqueue_intent','queue_failed']
    assert markers[0][0]==markers[1][0]


def test_concurrent_clients_keep_distinct_request_sequences():
    class Connection:
        def close(self): pass
    lane=module.RuntimeSupervisor.__new__(module.RuntimeSupervisor)
    lane.secret='00'*32;lane.commands=queue.Queue(2)
    lane.slots=type('Slots', (), {'release': lambda self: None})()
    request={'version': module.request.__globals__['VERSION'], 'op': 'prepare',
             'body': {'fixture': 'qwen'}}
    markers=[];guard=threading.Lock()
    def record(name, _, seq=0):
        with guard: markers.append((seq,name))
    with patch.object(module,'authenticate'),patch.object(module,'receive',return_value=request), \
         patch.object(module,'_prepare_phase',side_effect=record):
        clients=[threading.Thread(target=lane.client,args=(Connection(),)) for _ in range(2)]
        for client in clients:client.start()
        for client in clients:client.join(QWEN_PREPARE_RPC_SECONDS+1)
    assert all(not client.is_alive() for client in clients)
    sequences={seq for seq,_ in markers}
    assert len(sequences)==2
    assert all([name for observed,name in markers if observed==seq] ==
               ['enqueue_intent','server_prepare_response_wait_expired'] for seq in sequences)
