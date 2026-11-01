"""Trusted runtime evidence interface; never construct from worker messages."""
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class State(str, Enum):
    ALIVE = 'alive'
    STOPPED = 'stopped-and-reaped'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class Observation:
    runtime_id: str
    state: State


class Runtime(Protocol):
    """Exact identity observer. STOPPED requires independent termination proof.

    Implementations must bound every call; signal acknowledgement, PID absence,
    owner JSON and lock acquisition alone cannot produce STOPPED.
    """
    def observe(self, runtime_id: str, timeout: float) -> Observation: ...
    def terminate(self, runtime_id: str, timeout: float) -> None: ...
    def kill(self, runtime_id: str, timeout: float) -> None: ...
