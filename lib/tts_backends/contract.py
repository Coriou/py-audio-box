"""Runtime-independent audio and capability contract."""
from dataclasses import dataclass, field, asdict
from typing import Any
from numbers import Integral


class BackendError(ValueError):
    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code, self.details = code, details

    def as_dict(self):
        return {"code": self.code, "message": str(self), "details": self.details}


@dataclass(frozen=True)
class Capabilities:
    backend: str
    checkpoint: str
    modes: tuple[str, ...]
    text_direction: bool
    languages: tuple[str, ...] = ()

    def as_dict(self):
        return asdict(self)


@dataclass
class AudioResult:
    samples: Any
    sample_rate: int
    channels: int = 1
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        import numpy as np
        data = np.asarray(self.samples)
        if isinstance(self.sample_rate, bool) or isinstance(self.channels, bool) or not isinstance(self.sample_rate, Integral) or not isinstance(self.channels, Integral) or self.sample_rate <= 0 or self.channels <= 0:
            raise BackendError("invalid_audio", "Sample rate and channels must be positive")
        if data.ndim not in (1, 2) or (data.ndim == 1 and self.channels != 1) or (data.ndim == 2 and data.shape[1] != self.channels):
            raise BackendError("invalid_audio", "Audio must be samples by channels")
        if not np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.complexfloating):
            raise BackendError("invalid_audio", "Audio must contain real numeric samples")
        if not data.size or not np.isfinite(data).all():
            raise BackendError("invalid_audio", "Audio is empty or non-finite")
