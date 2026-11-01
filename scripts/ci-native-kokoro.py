#!/usr/bin/env python3
"""Bounded native Kokoro CPU preflight. No downloads, weights or worker admission."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def child(path):
    report = dict(status='failed', commit=os.environ.get('GITHUB_SHA', 'unknown'),
        runner_image=os.environ.get('ImageOS', 'unknown'), runner_version=os.environ.get('ImageVersion', 'unknown'),
        weights_loaded=False, installed_qualified=False, python=sys.version,
        lock_sha256=hashlib.sha256((ROOT/'requirements/kokoro.lock').read_bytes()).hexdigest())
    try:
        from worker.native_child import install_spawn_guard, SpawnRefused
        install_spawn_guard()
        from lib.tts_backends.kokoro_native import preflight
        report.update(preflight())
        import subprocess
        try:
            subprocess.Popen(['never-execute-kokoro-preflight'])
        except SpawnRefused:
            report['explicit_spawn_refused'] = True
        else:
            raise AssertionError('permanent_spawn_guard_lost')
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)[:2000],
            error_code=getattr(exc, 'code', None), cause_type=type(exc.__cause__).__name__ if exc.__cause__ else None,
            cause=str(exc.__cause__)[:2000] if exc.__cause__ else None)
        raise
    finally:
        save(path, report)


def run(output):
    from lib.tts_backends.kokoro_native import validate_host
    validate_host()
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('kokoro_owner', ROOT/'scripts/ci-apple-speech.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    observation = {}
    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'TMPDIR', 'GITHUB_SHA', 'ImageOS', 'ImageVersion')}
    with tempfile.TemporaryDirectory(prefix='kokoro-preflight-home-') as home:
        env.update(HOME=home, HF_HOME=home, HF_HUB_OFFLINE='1', OMP_NUM_THREADS='2')
        try:
            owner.owned_child([sys.executable, '-I', '-B', '-u', str(Path(__file__).resolve()),
                '--child-report', str(output.resolve()/'report.json')], env,
                output/'child.log', observation, seconds=60)
        except BaseException as exc:
            observation.update(error_type=type(exc).__name__, error=str(exc)[:1000])
            raise
        finally:
            save(output/'owner.json', observation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--child-report', type=Path)
    args = parser.parse_args()
    if args.child_report: child(args.child_report)
    elif args.output: run(args.output)
    else: parser.error('--output is required')
