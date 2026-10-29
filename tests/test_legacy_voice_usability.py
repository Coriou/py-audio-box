"""Standard-library focused tests; all voice assets are disposable temporary files.

Most caller checks compile selection functions from source AST. Regression checks
also execute complete app imports and repair parser/dispatch, stubbing only external
numerical/ML modules. This does not exercise inference, Docker or Linux acceptance.
"""
import argparse
import ast
import contextlib
import io
import os
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
from voices import VoiceRegistry, validate_slug


def caller(relative, name, **extra):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(Path=Path, VoiceRegistry=VoiceRegistry, validate_slug=validate_slug,
                     sys=sys, Any=object)
    namespace.update(extra)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@contextlib.contextmanager
def app_module(relative):
    """Execute the whole app; stub only unavailable external numerical/ML imports."""
    name = '_voice_usability_app'
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    external = {key: MagicMock(name=key) for key in ('numpy', 'soundfile', 'torch', 'torchaudio')}
    with patch.dict(sys.modules, {**external, name: module}):
        spec.loader.exec_module(module)
        yield module


class VoiceUsability(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)
        self.reg = VoiceRegistry(self.cache)
        self.reg.create('bruce', 'Bruce Wayne', {'type': 'file'})
        self.reg.source_clip('bruce').write_bytes(b'original source')
        self.resolve = caller('apps/voice-synth/voice-synth.py', 'resolve_voice')
        self.clone_ref = caller('apps/voice-clone/voice-clone.py', '_resolve_ref_audio')

    def snapshot(self):
        return {str(p.relative_to(self.cache)): p.read_bytes()
                for p in self.cache.rglob('*') if p.is_file()}

    def orphan(self):
        pd = self.reg.prompts_dir('bruce')
        pd.mkdir()
        pkl = pd / 'Qwen_0.6B_full_v1.pkl'
        pkl.write_bytes(b'not a pickle; must never be deserialized')
        ref = pkl.with_suffix('.ref.wav')
        ref.write_bytes(b'original per-prompt reference')
        self.reg.ref_wav('bruce').write_bytes(b'original current reference')
        meta = dict(engine='qwen3', checkpoint='Qwen/Qwen3-TTS-12Hz-0.6B-Base',
                    revision='a' * 40, schema_version=1, mode='full', cache_identity_version=1,
                    pickle_sha256=hashlib.sha256(pkl.read_bytes()).hexdigest(),
                    reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
                    reference_wav=ref.name)
        pkl.with_suffix('.meta.json').write_text(json.dumps(meta))
        data = self.reg.load('bruce')
        data['prompts'] = {pkl.stem: f'prompts/{pkl.name}'}
        data['tones'] = {'neutral': pkl.stem}
        data['aliases'] = ['batman']
        backup = self.cache / 'trusted-backup.json'
        backup.write_text(json.dumps(data))
        self.reg.voice_json('bruce').unlink()
        return backup, pkl

    def test_case_display_alias_and_prefix_in_both_callers(self):
        self.reg.add_alias('BRUCE', 'Batman')
        pd = self.reg.prompts_dir('bruce'); pd.mkdir()
        pkl = pd / 'CaseSensitive.pkl'; pkl.write_bytes(b'prompt')
        for name in ('BRUCE', 'BrUcE WaYnE', 'BATMAN', 'BrU'):
            with self.subTest(name=name):
                self.assertEqual(self.reg.resolve_name(name), 'bruce')
                result = self.resolve(name, self.cache)
                self.assertEqual(result['named_slug'], 'bruce')
                self.assertEqual(Path(result['prompt_path']), pkl)
                args = SimpleNamespace(voice=name)
                self.assertEqual(self.clone_ref(args, self.cache), self.reg.source_clip('bruce'))
                self.assertEqual(args.voice, 'bruce')

    def test_ambiguous_prefix_and_display_name_fail(self):
        self.reg.create('bruno', 'Bruce Wayne', {})
        for name in ('Br', 'BRUCE WAYNE'):
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                self.resolve(name, self.cache)
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                self.clone_ref(SimpleNamespace(voice=name), self.cache)
        self.assertEqual(self.reg.resolve_name('BRUCE'), 'bruce')

    def test_corrupt_registry_never_falls_back_to_legacy_in_actual_consumer(self):
        pd = self.reg.prompts_dir('bruce'); pd.mkdir()
        (pd / 'named.pkl').write_bytes(b'named voice')
        legacy = self.cache / 'voice-clone/prompts'; legacy.mkdir(parents=True)
        (legacy / 'bruce.pkl').write_bytes(b'different legacy voice')
        bad = self.reg.voice_dir('broken'); bad.mkdir()
        record = bad / 'voice.json'
        valid = dict(slug='broken', created_at='2000', source={})
        corruptions = [None, [], {k: v for k, v in valid.items() if k != 'slug'}]
        corruptions += [dict(valid, slug=value) for value in
                        (None, 123, [], {}, '../bruce', 'BROKEN', ' broken ', 'bruce')]
        corruptions += [dict(valid, **{field: value}) for field, value in
                        (('created_at', []), ('display_name', {}), ('aliases', [123]),
                         ('schema_version', 999))]
        with app_module('apps/voice-synth/voice-synth.py') as synth:
            self.assertEqual(synth.resolve_voice('BRUCE', self.cache)['prompt_path'], str(pd / 'named.pkl'))
            for raw in [json.dumps(data) for data in corruptions] + ['{invalid json']:
                with self.subTest(raw=raw):
                    record.write_text(raw)
                    before = self.snapshot()
                    with self.assertRaisesRegex(ValueError, 'Invalid voice registry metadata.*broken'):
                        synth.resolve_voice('BRUCE', self.cache)
                    with self.assertRaisesRegex(ValueError, 'Invalid voice registry metadata'):
                        self.reg.load('broken')
                    self.assertEqual(self.snapshot(), before)
            record.unlink()
            self.assertEqual(synth.resolve_voice('BRUCE', self.cache)['prompt_path'], str(pd / 'named.pkl'))

    def test_ordinary_lookup_miss_still_selects_legacy(self):
        legacy = self.cache / 'voice-clone/prompts'; legacy.mkdir(parents=True)
        pkl = legacy / 'legacy.pkl'; pkl.write_bytes(b'legacy prompt')
        self.assertEqual(self.resolve('LEGACY', self.cache)['prompt_path'], str(pkl))

    def test_alias_collision_and_path_rejected(self):
        self.reg.create('john', 'John', {})
        for alias in ('JOHN', '../john'):
            with self.assertRaises(ValueError):
                self.reg.add_alias('bruce', alias)
        self.reg.voice_dir('orphan').mkdir()
        with self.assertRaises(ValueError):
            self.reg.add_alias('bruce', 'orphan')

    def test_old_collision_is_never_silently_resolved(self):
        self.reg.add_alias('bruce', 'hero')
        self.reg.create('hero', 'Hero', {})
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            self.reg.resolve_name('HERO')

    def test_legacy_ids_casefold_but_preserve_path_and_refuse_collision(self):
        pd = self.cache / 'voice-clone/prompts'; pd.mkdir(parents=True)
        pkl = pd / 'Qwen_0.6B_Bruce.pkl'; pkl.write_bytes(b'original')
        for name in ('QWEN_0.6b_BRUCE', 'qwen_0.6'):
            self.assertEqual(Path(self.resolve(name, self.cache)['prompt_path']), pkl)
        self.assertEqual(Path(self.resolve(str(pkl), self.cache)['prompt_path']), pkl)
        with self.assertRaises(SystemExit):
            self.resolve(str(pkl.parent / 'Missing.pkl'), self.cache)
        (pd / 'qwen_0.6b_bruce.pkl').write_bytes(b'other')
        if len(list(pd.glob('*.pkl'))) == 2:
            with self.assertRaises(SystemExit):
                self.resolve('Qwen_0.6B_Bruce', self.cache)
        else:
            # Case collisions cannot be constructed on this filesystem.
            self.assertEqual(self.resolve('QWEN_0.6B_BRUCE', self.cache)['prompt_path'], str(pkl))

    def test_tones_use_canonical_voice_and_keep_case_sensitive_prompt_stem(self):
        self.reg.add_alias('bruce', 'hero')
        pd = self.reg.prompts_dir('bruce'); pd.mkdir()
        pkl = pd / 'Full_0.6B.pkl'; pkl.write_bytes(b'prompt')
        data = self.reg.load('bruce'); data['tones'] = {'calm': pkl.stem}
        self.reg._save('bruce', data)
        resolved = self.resolve('HERO', self.cache, tone='calm')
        self.assertEqual(resolved['named_slug'], 'bruce')
        self.assertEqual(resolved['prompt_path'], str(pkl))

    def test_custom_voice_alias_keeps_profile_and_defaults(self):
        self.reg.register_custom_voice('announcer', model='custom-model', speaker='Ryan',
                                       instruct_default='Calm', language_default='English')
        self.reg.add_alias('announcer', 'radio')
        resolved = self.resolve('RADIO', self.cache)
        self.assertEqual(resolved['voice_id'], 'announcer')
        self.assertEqual(resolved['engine_mode'], 'custom_voice')
        self.assertEqual(resolved['speaker'], 'Ryan')
        self.assertEqual(resolved['model'], 'custom-model')
        self.assertIsNone(resolved['prompt_path'])

    def test_management_alias_is_exact_and_never_uses_destructive_prefix(self):
        relative = 'apps/voice-synth/voice-synth.py'
        self.reg.add_alias('bruce', 'hero')
        delete = caller(relative, 'cmd_delete_voice', _voice_reg=lambda args: self.reg)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            delete(SimpleNamespace(voice='HE', yes=True))
        self.assertTrue(self.reg.exists('bruce'))
        rename = caller(relative, 'cmd_rename_voice', _voice_reg=lambda args: self.reg)
        with contextlib.redirect_stdout(io.StringIO()):
            rename(SimpleNamespace(old_name='HERO', new_name='batman'))
        self.assertEqual(self.reg.resolve_name('HERO'), 'batman')
        with contextlib.redirect_stdout(io.StringIO()):
            delete(SimpleNamespace(voice='HERO', yes=True))
        self.assertFalse(self.reg.exists('batman'))

    def test_cli_preview_apply_and_alias_dispatch_without_runtime_setup(self):
        backup, _ = self.orphan()
        relative = 'apps/voice-clone/voice-clone.py'
        common = caller(relative, '_add_common', argparse=argparse, os=os,
                        DEFAULT_TTS_MODEL='unused', DEFAULT_WHISPER='unused', QWEN3_LANGUAGES=[])
        parser = caller(relative, 'build_parser', argparse=argparse, _add_common=common,
                        GENERATION_PROFILE_CHOICES=[])()
        main = caller(relative, 'main', build_parser=lambda: parser, json=json)
        command = ['voice-clone', 'repair-registry', '--voice', 'BRUCE',
                   '--metadata', str(backup), '--cache', str(self.cache)]
        before = self.snapshot()
        with patch.object(sys, 'argv', command), contextlib.redirect_stdout(io.StringIO()) as out:
            main()
        self.assertTrue(json.loads(out.getvalue())['dry_run'])
        self.assertEqual(before, self.snapshot())
        with patch.object(sys, 'argv', command + ['--apply']), contextlib.redirect_stdout(io.StringIO()) as out:
            main()
        self.assertTrue(json.loads(out.getvalue())['restored'])
        with patch.object(sys, 'argv', ['voice-clone', 'add-alias', '--voice', 'BRUCE',
                          '--alias', 'Hero', '--cache', str(self.cache)]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            main()
        self.assertIn('hero', json.loads(out.getvalue())['aliases'])
        # main's runtime symbols were not supplied: reaching model/hub setup would fail.

    def test_repair_preview_and_apply_preserve_all_existing_bytes(self):
        backup, pkl = self.orphan()
        before = self.snapshot()
        with patch('pickle.load', side_effect=AssertionError('unpickle forbidden')), \
             patch('pickle.loads', side_effect=AssertionError('unpickle forbidden')):
            self.assertTrue(self.reg.repair_orphan('BRUCE', backup)['dry_run'])
            self.assertEqual(before, self.snapshot())
            self.assertTrue(self.reg.repair_orphan('bruce', backup, dry_run=False)['restored'])
        after = self.snapshot()
        self.assertEqual(after.pop('voices/bruce/voice.json'), backup.read_bytes())
        self.assertEqual(before, after)
        self.assertEqual(self.reg.resolve_name('BATMAN'), 'bruce')
        self.assertEqual(self.reg.matching_prompt('bruce', checkpoint='Qwen/Qwen3-TTS-12Hz-0.6B-Base',
                         revision='a'*40, schema_version=1), pkl)
        self.assertIsNone(self.reg.matching_prompt('bruce', checkpoint='Qwen/Qwen3-TTS-12Hz-1.7B-Base',
                          revision='a'*40, schema_version=1))

    def test_actual_repair_cli_refuses_non_clone_checkpoints_without_publication(self):
        backup, pkl = self.orphan()
        sidecar = pkl.with_suffix('.meta.json')
        meta = json.loads(sidecar.read_text())
        local = self.cache / 'local-Base'; local.mkdir()
        config = local / 'config.json'
        cases = [(f'Qwen/Qwen3-TTS-12Hz-{size}-{mode}', None)
                 for size, mode in (('0.6B', 'CustomVoice'), ('1.7B', 'CustomVoice'),
                                    ('1.7B', 'VoiceDesign'), ('0.6B', 'Unknown'))]
        cases += [(str(local), dict(model_type='qwen3_tts', tts_model_size='0b6',
                                   tts_model_type=mode))
                  for mode in ('custom_voice', 'voice_design', 'unknown')]
        with app_module('apps/voice-clone/voice-clone.py') as clone, \
             patch('pickle.load', side_effect=AssertionError('unpickle forbidden')), \
             patch('pickle.loads', side_effect=AssertionError('unpickle forbidden')):
            for checkpoint, local_config in cases:
                if local_config is not None:
                    config.write_text(json.dumps(local_config))
                meta['checkpoint'] = checkpoint
                sidecar.write_text(json.dumps(meta))
                before = self.snapshot()
                for apply in ([], ['--apply']):
                    with self.subTest(checkpoint=checkpoint, config=local_config, apply=apply):
                        args = ['voice-clone', 'repair-registry', '--voice', 'bruce',
                                '--metadata', str(backup), '--cache', str(self.cache), *apply]
                        with patch.object(sys, 'argv', args), \
                             contextlib.redirect_stdout(io.StringIO()) as out, \
                             contextlib.redirect_stderr(io.StringIO()) as err, \
                             self.assertRaises(SystemExit) as raised:
                            clone.main()
                        self.assertEqual(raised.exception.code, 2)
                        self.assertEqual(out.getvalue(), '')
                        self.assertRegex(err.getvalue(), 'does not support clone|capabilities are unknown')
                        self.assertNotIn('Traceback', err.getvalue())
                        self.assertFalse(self.reg.exists('bruce'))
                        self.assertEqual(self.snapshot(), before)

    def test_repair_accepts_locally_identified_base_checkpoint(self):
        backup, pkl = self.orphan()
        local = self.cache / 'local-checkpoint'; local.mkdir()
        (local / 'config.json').write_text(json.dumps(dict(model_type='qwen3_tts',
                    tts_model_size='0b6', tts_model_type='base')))
        sidecar = pkl.with_suffix('.meta.json')
        meta = json.loads(sidecar.read_text()); meta['checkpoint'] = str(local)
        sidecar.write_text(json.dumps(meta))
        before = self.snapshot()
        self.assertTrue(self.reg.repair_orphan('bruce', backup)['dry_run'])
        self.assertEqual(before, self.snapshot())
        self.assertTrue(self.reg.repair_orphan('bruce', backup, dry_run=False)['restored'])
        after = self.snapshot()
        self.assertEqual(after.pop('voices/bruce/voice.json'), backup.read_bytes())
        self.assertEqual(after, before)

    def test_repair_refuses_missing_backup_evidence_and_never_mutates(self):
        for damage in ('pickle', 'reference', 'meta', 'model', 'revision', 'schema', 'extra',
                       'traversal', 'slug', 'tone', 'symlink'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as td:
                original = self.cache, self.reg
                self.cache = Path(td); self.reg = VoiceRegistry(self.cache)
                self.reg.create('bruce', 'Bruce', {})
                backup, pkl = self.orphan()
                sidecar = pkl.with_suffix('.meta.json')
                meta = json.loads(sidecar.read_text())
                data = json.loads(backup.read_text())
                if damage == 'pickle': pkl.write_bytes(b'tampered')
                elif damage == 'reference': pkl.with_suffix('.ref.wav').write_bytes(b'tampered')
                elif damage == 'meta': sidecar.unlink()
                elif damage == 'model': meta.pop('checkpoint'); sidecar.write_text(json.dumps(meta))
                elif damage == 'revision': meta['revision'] = 'main'; sidecar.write_text(json.dumps(meta))
                elif damage == 'schema': meta['schema_version'] = 999; sidecar.write_text(json.dumps(meta))
                elif damage == 'extra': (pkl.parent / 'unknown.pkl').write_bytes(b'unknown')
                elif damage == 'traversal': data['prompts'][pkl.stem] = '../outside.pkl'
                elif damage == 'slug': data['slug'] = 'john'
                elif damage == 'tone': data['tones'] = {'neutral': 'missing'}
                else:
                    raw = pkl.read_bytes(); pkl.unlink()
                    outside = self.cache / 'outside.pkl'; outside.write_bytes(raw); pkl.symlink_to(outside)
                backup.write_text(json.dumps(data))
                before = self.snapshot()
                for dry_run in (True, False):
                    with self.assertRaises((ValueError, OSError)):
                        self.reg.repair_orphan('bruce', backup, dry_run=dry_run)
                    self.assertEqual(self.snapshot(), before)
                self.cache, self.reg = original

    def test_repair_refuses_existing_record(self):
        backup, _ = self.orphan()
        self.reg.voice_json('bruce').write_bytes(b'new live registration')
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.reg.repair_orphan('bruce', backup, dry_run=False)
        self.assertEqual(before, self.snapshot())

    def test_failed_atomic_publication_leaves_orphan_intact(self):
        backup, _ = self.orphan()
        before = self.snapshot()
        with patch('voices.os.link', side_effect=OSError('injected publication failure')):
            with self.assertRaises(OSError):
                self.reg.repair_orphan('bruce', backup, dry_run=False)
        self.assertEqual(before, self.snapshot())


if __name__ == '__main__':
    unittest.main()
