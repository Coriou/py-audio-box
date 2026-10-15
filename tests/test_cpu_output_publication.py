import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
spec = importlib.util.spec_from_file_location('cpu_publication', ROOT / 'apps/voice-cpu/voice-cpu.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def test_concurrent_destination_never_overwritten(tmp_path):
    output = tmp_path / 'take.wav'
    output.write_bytes(b'another writer')
    result = types.SimpleNamespace(samples=np.zeros(240), sample_rate=24000)
    with pytest.raises(FileExistsError):
        app.publish_take(output, result, {'status': 'ok'})
    assert output.read_bytes() == b'another writer'
    assert not list(tmp_path.glob('.take-*'))


def test_metadata_collision_rolls_back_only_owned_audio(tmp_path):
    output = tmp_path / 'take.wav'
    metadata = output.with_suffix('.meta.json')
    metadata.write_text('another manifest')
    result = types.SimpleNamespace(samples=np.zeros(240), sample_rate=24000)
    with pytest.raises(FileExistsError):
        app.publish_take(output, result, {'status': 'ok'})
    assert not output.exists()
    assert metadata.read_text() == 'another manifest'
