import concurrent.futures
import hashlib
import json
import sys
import wave
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from voices import VoiceRegistry


def archive(path, entries):
    with zipfile.ZipFile(path, 'w') as z:
        for name, value in entries:
            z.writestr(name, value)
    return path


def metadata(slug='narrator'):
    return json.dumps({'slug': slug, 'created_at': 'today', 'source': {}, 'prompts': {}})


@pytest.mark.parametrize('extra', [
    ('other/voice.json', metadata('other')), ('narrator/../evil', 'bad'),
    ('/narrator/evil', 'bad'), ('narrator/prompts/evil.pkl', 'bad'),
    ('narrator\\evil', 'bad'), ('narrator/voice.json', metadata()),
])
def test_archive_rejects_unsafe_entries_without_changes(tmp_path, extra):
    reg = VoiceRegistry(tmp_path / 'cache')
    reg.create('narrator', 'Original', {})
    original = reg.voice_json('narrator').read_bytes()
    z = archive(tmp_path / 'in.zip', [('narrator/voice.json', metadata()), extra])
    with pytest.raises(ValueError):
        reg.import_zip(z, force=True)
    assert reg.voice_json('narrator').read_bytes() == original


def test_portable_export_import_and_backup(tmp_path):
    reg = VoiceRegistry(tmp_path / 'cache')
    reg.create('narrator', 'Original', {'path': '/private/reference', 'type': 'file'})
    reg.prompts_dir('narrator').mkdir()
    (reg.prompts_dir('narrator') / 'unsafe.pkl').write_bytes(b'not portable')
    with wave.open(str(reg.ref_wav('narrator')), 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
        wav.writeframes(b'\0\0' * 24)
    path = reg.export_zip('narrator', tmp_path / 'out.zip')
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist()) == {'narrator/voice.json', 'narrator/ref.wav'}
        assert 'path' not in json.loads(z.read('narrator/voice.json'))['source']
    reg.import_zip(path, force=True)
    assert not reg.best_prompt('narrator')
    backups = list((reg.root / '.backups').glob('narrator-*'))
    assert len(backups) == 1
    assert (backups[0] / 'prompts/unsafe.pkl').exists()


def test_symlink_archive_rejected(tmp_path):
    zpath = tmp_path / 'bad.zip'
    with zipfile.ZipFile(zpath, 'w') as z:
        z.writestr('narrator/voice.json', metadata())
        entry = zipfile.ZipInfo('narrator/ref.wav')
        entry.create_system = 3
        entry.external_attr = 0o120777 << 16
        z.writestr(entry, '/etc/passwd')
    with pytest.raises(ValueError):
        VoiceRegistry(tmp_path / 'cache').import_zip(zpath)


def test_concurrent_mutations_keep_all_fields(tmp_path):
    reg = VoiceRegistry(tmp_path)
    reg.create('narrator', 'Original', {})
    def update(i):
        VoiceRegistry(tmp_path).register_custom_voice('builtin', model='model', speaker='Ryan', tone=f't{i}', tone_instruct=f'Instruction {i}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(update, range(24)))
    assert len(reg.load('builtin')['custom_voice']['tones']) == 24


def test_migration_dry_run_and_backup(tmp_path):
    reg = VoiceRegistry(tmp_path)
    reg.create('narrator', 'Original', {})
    data = reg.load('narrator'); data.pop('schema_version')
    reg.voice_json('narrator').write_text(json.dumps(data))
    original = reg.voice_json('narrator').read_bytes()
    assert reg.migrate('narrator')['changed']
    assert reg.voice_json('narrator').read_bytes() == original
    reg.migrate('narrator', dry_run=False)
    assert reg.load('narrator')['schema_version'] == 2
    assert next(reg.voice_dir('narrator').glob('*.backup.json')).read_bytes() == original


def test_prompt_identity_without_unpickling(tmp_path):
    reg = VoiceRegistry(tmp_path)
    path = tmp_path / 'cache.pkl'; path.write_bytes(b'not a pickle')
    meta = dict(engine='qwen3', checkpoint='checkpoint', revision='sha', cache_identity_version=1,
                schema_version=1, mode='full', reference_sha256='ref', pickle_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.meta.json').write_text(json.dumps(meta))
    assert reg.compatible_prompt(path, checkpoint='checkpoint', revision='sha', reference_hash='ref')
    assert not reg.compatible_prompt(path, checkpoint='other', revision='sha')
    assert not reg.compatible_prompt(path, checkpoint='checkpoint', revision='other')
    path.write_bytes(b'tampered')
    assert not reg.compatible_prompt(path, checkpoint='checkpoint', revision='sha')


def test_archive_limits_and_invalid_wav_leave_original(tmp_path):
    reg = VoiceRegistry(tmp_path / 'cache')
    reg.create('narrator', 'Original', {})
    original = reg.voice_json('narrator').read_bytes()
    for entries in (
        [('narrator/voice.json', metadata()), ('narrator/ref.wav', b'not audio')],
        [('narrator/voice.json', ' ' * (1024 * 1024 + 1))],
        [('narrator/voice.json', metadata())] * 65,
    ):
        zpath = archive(tmp_path / 'bad.zip', entries)
        with pytest.raises((ValueError, wave.Error)):
            reg.import_zip(zpath, force=True)
        assert reg.voice_json('narrator').read_bytes() == original
        assert not list(reg.root.glob('.import-*'))


def test_tone_cache_uses_own_reference_after_other_tone_registered(tmp_path):
    reg = VoiceRegistry(tmp_path)
    reg.create('narrator', 'Narrator', {})
    prompt = tmp_path / 'local.pkl'; prompt.write_bytes(b'trusted local placeholder')
    source = tmp_path / 'local.wav'
    paths = []
    for tone, content in [('neutral', b'first reference'), ('excited', b'second reference')]:
        source.write_bytes(content)
        reg.update_ref('narrator', source, {})
        meta = dict(engine='qwen3', checkpoint='checkpoint', revision='sha', cache_identity_version=1,
                    schema_version=1, mode='full', reference_sha256=hashlib.sha256(content).hexdigest(),
                    pickle_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest())
        paths.append(reg.register_prompt('narrator', tone, prompt, meta, tone=tone))
    neutral = paths[0]
    reference = reg.prompt_reference(neutral)
    assert reference.read_bytes() == b'first reference'
    assert reg.compatible_prompt(neutral, checkpoint='checkpoint', revision='sha',
                                 reference_hash=hashlib.sha256(reference.read_bytes()).hexdigest())
    reference.write_bytes(b'tampered selected reference')
    assert not reg.compatible_prompt(neutral, checkpoint='checkpoint', revision='sha',
                                     reference_hash=hashlib.sha256(reference.read_bytes()).hexdigest())
