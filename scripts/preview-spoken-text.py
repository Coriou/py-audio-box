#!/usr/bin/env python3
"""Read a JSON request from stdin or one file; print canonical preview JSON."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.spoken_text import canonical_json, render


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON field')
        result[key] = value
    return result


def main():
    if len(sys.argv) > 2:
        raise ValueError('usage: preview-spoken-text.py [request.json]')
    if len(sys.argv) == 2:
        with Path(sys.argv[1]).open('rb') as source:
            raw = source.read(65537)
    else:
        raw = sys.stdin.buffer.read(65537)
    if len(raw) > 65536:
        raise ValueError('JSON exceeds 65536 bytes')
    request = json.loads(raw, object_pairs_hook=unique_object)
    print(canonical_json(render(request)))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        print('preview error: ' + str(exc), file=sys.stderr)
        sys.exit(2)
