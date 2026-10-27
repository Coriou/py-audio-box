#!/usr/bin/env python3
"""Reproduce the narrowly patched native wheel from exact public upstream bytes.

No network, installation, model loading or subprocess execution occurs here.
The two modified upstream files remain Apache-2.0; application source is AGPL.
"""
import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile

UPSTREAM_SHA256 = '11a290d8dabc7ef91a90c54478c8ab19b3edb1d85c0882313721892bdc4af15d'
FILENAME = 'qwen_tts-0.1.1-1native12hz-py3-none-any.whl'
DIST = 'qwen_tts-0.1.1.dist-info/'
CORE = 'qwen_tts/core/__init__.py'
TOKENIZER = 'qwen_tts/inference/qwen3_tts_tokenizer.py'
ORIGINALS = {
    CORE: '1b380d9de843b6d585d938c339d066136567ca7125412674234204af4386679e',
    TOKENIZER: 'ac2d855022a1bd21d33ab7b267ec952eef71f81d3f8d969a306139ff1a929515',
}
MARK = '# Modified for py-audio-box native wheel build 1native12hz: defer unused 25 Hz model imports.\n'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('upstream_patch_context_changed')
    return source.replace(old, new, 1)


def patched_members(raw):
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA256:
        raise ValueError('upstream_wheel_sha256_mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as wheel:
        members = {entry.filename: wheel.read(entry) for entry in wheel.infolist()}
    for name, digest in ORIGINALS.items():
        if hashlib.sha256(members[name]).hexdigest() != digest:
            raise ValueError('upstream_file_sha256_mismatch')
    core = members[CORE].decode()
    core = replace_once(core,
        'from .tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Model\n', '')
    core += '''

def __getattr__(name):
    if name == "Qwen3TTSTokenizerV1Model":
        from .tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Model
        globals()[name] = Qwen3TTSTokenizerV1Model
        return Qwen3TTSTokenizerV1Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
'''
    tokenizer = members[TOKENIZER].decode()
    tokenizer = replace_once(tokenizer, '    Qwen3TTSTokenizerV1Model,\n', '')
    tokenizer = replace_once(tokenizer, '''        AutoConfig.register("qwen3_tts_tokenizer_25hz", Qwen3TTSTokenizerV1Config)
        AutoModel.register(Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model)

        AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
        AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)
''', '''        AutoConfig.register("qwen3_tts_tokenizer_25hz", Qwen3TTSTokenizerV1Config)
        AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if config.model_type == "qwen3_tts_tokenizer_25hz":
            from ..core import Qwen3TTSTokenizerV1Model
            AutoModel.register(Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model)
        else:
            AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)
''')
    for name, source in ((CORE, core), (TOKENIZER, tokenizer)):
        members[name] = (MARK + source).encode()
    members[DIST + 'WHEEL'] += b'Build: 1native12hz\n'
    members[DIST + 'NOTICE'] = (
        'Qwen-TTS 0.1.1, Copyright 2026 The Alibaba Qwen team. Apache-2.0.\n'
        'Upstream: https://github.com/QwenLM/Qwen3-TTS\n'
        'Modified by py-audio-box: build 1native12hz lazily imports the unused\n'
        '25 Hz tokenizer model. The 12 Hz route no longer imports PySox.\n'
        'Only core/__init__.py and inference/qwen3_tts_tokenizer.py changed.\n'
        'Original LICENSE and package metadata/dependency requirements retained.\n'
    ).encode()
    record = io.StringIO(newline='')
    writer = csv.writer(record, lineterminator='\n')
    for name, content in sorted(members.items()):
        if name == DIST + 'RECORD': continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b'=').decode()
        writer.writerow((name, 'sha256=' + digest, str(len(content))))
    writer.writerow((DIST + 'RECORD', '', ''))
    members[DIST + 'RECORD'] = record.getvalue().encode()
    return members


def build(upstream, output):
    if upstream.stat().st_size > 1024**2:
        raise ValueError('upstream_wheel_size_limit')
    members = patched_members(upstream.read_bytes())
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_STORED) as wheel:
        for name, content in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            wheel.writestr(info, content)
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.name != FILENAME:
        parser.error('output must be named ' + FILENAME)
    print(build(args.upstream, args.output))
