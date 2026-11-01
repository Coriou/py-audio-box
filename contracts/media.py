"""True tiny synthetic media for E1 examples; no synthesis or rendering."""
import hashlib
import io
import numpy as np
import soundfile as sf


def synthetic_media(frames, rate, channels, format, subtype):
    buffer=io.BytesIO()
    sf.write(buffer,np.zeros((frames,channels)),rate,format=format,subtype=subtype)
    data=buffer.getvalue()
    info=sf.info(io.BytesIO(data))
    return data,dict(sha256=hashlib.sha256(data).hexdigest(),frames=info.frames,sample_rate=info.samplerate,channels=info.channels,media_format=info.format.lower(),media_subtype=info.subtype)
