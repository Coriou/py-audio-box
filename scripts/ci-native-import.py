#!/usr/bin/env python3
"""Fresh real locked bootstrap under the product guard; never acquires weights."""
import argparse
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import platform
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WHEEL = ROOT / 'requirements/wheels/qwen_tts-0.1.1-1native12hz-py3-none-any.whl'
WHEEL_SHA256 = '96e5d3dfdcf2a86bcb59d9a68de7ef51bd6f591a63cdae816a7c1da4cad31606'


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def child(report_path):
    report = dict(status='failed', weights_loaded=False, runtime_qualified=False,
        commit=os.environ.get('GITHUB_SHA', 'unknown'),
        runner_image=os.environ.get('ImageOS', 'unknown'),
        runner_version=os.environ.get('ImageVersion', 'unknown'),
        python=sys.version, stage='artifact')
    try:
        if hashlib.sha256(WHEEL.read_bytes()).hexdigest() != WHEEL_SHA256:
            raise ValueError('native_qwen_artifact_hash')
        distribution = metadata.distribution('qwen-tts')
        if distribution.version != '0.1.1':
            raise ValueError('native_qwen_version')
        # Verify actual installed dependency bytes, not just a package version.
        with zipfile.ZipFile(WHEEL) as artifact:
            for name in artifact.namelist():
                if name.endswith('/RECORD'): continue  # pip rewrites installed RECORD.
                if distribution.locate_file(name).read_bytes() != artifact.read(name):
                    raise ValueError('native_qwen_installed_artifact_mismatch: ' + name)
        report['artifact_sha256'] = WHEEL_SHA256
        from worker import native_child
        native_child.install_spawn_guard()
        report['stage'] = 'product_bootstrap'
        from lib.tts import _qwen_model_class
        model_class = _qwen_model_class()
        from qwen_tts import Qwen3TTSTokenizer
        from qwen_tts.core import Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model
        from transformers import AutoModel
        if not callable(model_class.from_pretrained) or not callable(Qwen3TTSTokenizerV2Model.forward):
            raise ValueError('real_loader_classes_missing')
        report['stage'] = '12hz_dispatch_without_weights'
        with tempfile.TemporaryDirectory(prefix='native-import-config-') as private:
            root = Path(private).resolve()
            # Config-only fixture: no model/voice data and no model instantiation.
            save(root/'config.json', {'model_type': 'qwen3_tts_tokenizer_12hz'})
            save(root/'preprocessor_config.json', {'feature_extractor_type': 'WhisperFeatureExtractor'})
            try:
                Qwen3TTSTokenizer.from_pretrained(str(root), local_files_only=True)
            except OSError as exc:
                if 'no file named' not in str(exc): raise
                report['expected_missing_weights'] = str(exc)
            else:
                raise AssertionError('config_only_fixture_must_not_construct_model')
        if AutoModel._model_mapping[Qwen3TTSTokenizerV2Config] is not Qwen3TTSTokenizerV2Model:
            raise ValueError('12hz_dispatch_not_registered')
        if 'sox' in sys.modules or any(n.startswith('qwen_tts.core.tokenizer_25hz.modeling') for n in sys.modules):
            raise ValueError('unused_25hz_or_pysox_imported')
        report['stage'] = 'guard_retained'
        import subprocess
        try:
            subprocess.Popen(['never-execute-native-import-regression'])
        except native_child.SpawnRefused as exc:
            report['explicit_spawn_refused'] = str(exc)
        else:
            raise AssertionError('permanent_spawn_guard_lost')
        if not native_child._guard_installed:
            raise ValueError('guard_not_installed')
        report.update(status='passed', stage='complete', pysox_imported=False,
            legacy_25hz_model_imported=False, real_12hz_dispatch=True,
            versions={n: metadata.version(n) for n in ('qwen-tts', 'torch', 'torchaudio',
                'transformers', 'tokenizers', 'accelerate', 'librosa', 'soundfile', 'sox')})
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)[:4000])
        raise
    finally:
        save(report_path, report)


def run(output):
    if (platform.system(), platform.machine(), sys.version_info[:2]) != ('Darwin', 'arm64', (3, 11)):
        raise ValueError('native_import_check_requires_Darwin_arm64_Python311')
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('apple_speech_owner', ROOT/'scripts/ci-apple-speech.py')
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    observation = {}
    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'TMPDIR', 'GITHUB_SHA', 'ImageOS', 'ImageVersion')}
    with tempfile.TemporaryDirectory(prefix='native-import-home-') as home:
        env.update(HOME=home, HF_HOME=home, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
            PYTORCH_ENABLE_MPS_FALLBACK='0')
        try:
            owner.owned_child([sys.executable, '-I', '-B', str(Path(__file__).resolve()),
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
    if args.child_report:
        child(args.child_report)
    elif args.output:
        run(args.output)
    else:
        parser.error('--output is required')
