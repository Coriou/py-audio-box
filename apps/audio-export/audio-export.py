#!/usr/bin/env python3
"""Create a separate PCM24 WAV master with actual resampling via FFmpeg."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def export_master(source: Path, dest: Path, *, rate: int, channels: int,
                  loudness: float | None = None) -> Path:
    source, dest = Path(source), Path(dest)
    if not source.is_file():
        raise ValueError('Source audio does not exist')
    if dest.exists() or source.resolve() == dest.resolve():
        raise ValueError('Destination must be a new file; originals are preserved')
    if rate not in (24000, 44100, 48000, 96000) or channels not in (1, 2):
        raise ValueError('Unsupported output rate or channel count')
    if loudness is not None and not -70 <= loudness <= -5:
        raise ValueError('Loudness target must be between -70 and -5 LUFS')
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.master-', suffix='.wav', dir=dest.parent)
    os.close(fd)
    try:
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source), '-map', '0:a:0', '-vn']
        if loudness is not None:
            command += ['-af', f'loudnorm=I={loudness}:TP=-1.0:LRA=11']
        command += ['-ar', str(rate), '-ac', str(channels), '-c:a', 'pcm_s24le', '-f', 'wav', temporary]
        subprocess.run(command, check=True, timeout=300)
        # Link is atomic and refuses a destination created by a concurrent writer.
        os.link(temporary, dest)
    finally:
        os.unlink(temporary)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rate', required=True, type=int, choices=(24000, 44100, 48000, 96000))
    parser.add_argument('--channels', required=True, type=int, choices=(1, 2))
    parser.add_argument('--loudness', type=float, help='Optional single-pass FFmpeg loudnorm target in LUFS; verify delivery loudness separately')
    args = parser.parse_args()
    print(export_master(args.input, args.output, rate=args.rate, channels=args.channels, loudness=args.loudness))


if __name__ == '__main__':
    main()
