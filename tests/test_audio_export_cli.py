import importlib.util
from pathlib import Path
import shutil
import wave

import pytest

spec = importlib.util.spec_from_file_location('audio_export', Path(__file__).resolve().parents[1] / 'apps/audio-export/audio-export.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg required')
def test_actual_resample_pcm24_preserves_duration_and_original(tmp_path):
    source = tmp_path / 'source.wav'
    with wave.open(str(source), 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
        wav.writeframes(b'\0\0' * 24000)
    original = source.read_bytes()
    dest = tmp_path / 'master.wav'
    module.export_master(source, dest, rate=48000, channels=2)
    import soundfile as sf
    result = sf.info(dest)
    assert result.samplerate == 48000 and result.channels == 2
    assert result.frames == 48000 and result.subtype == 'PCM_24'
    assert source.read_bytes() == original
    with pytest.raises(ValueError):
        module.export_master(source, dest, rate=48000, channels=2)
