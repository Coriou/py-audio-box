import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def test_local_prompt_cache_invalidates_revision_reference_and_transcript(tmp_path):
    spec = importlib.util.spec_from_file_location('voice_clone_identity', ROOT / 'apps/voice-clone/voice-clone.py')
    vc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vc)
    reference = tmp_path / 'reference.wav'; reference.write_bytes(b'local audio placeholder')
    ref = SimpleNamespace(ref_hash='stable', segment=reference, transcript='Hello', ref_language='English',
                          ref_language_prob=1.0, seg_start=0.0, seg_end=1.0)
    class Model:
        name_or_path = 'Qwen/checkpoint'
        _audio_box_checkpoint = 'Qwen/checkpoint'
        _audio_box_revision = 'first'
        calls = 0
        def create_voice_clone_prompt(self, **kwargs):
            self.calls += 1
            return {'generation': self.calls}
    model = Model()
    first, path = vc.build_voice_clone_prompt(model, ref, tmp_path, False, False)
    assert vc.build_voice_clone_prompt(model, ref, tmp_path, False, False)[0] == first
    assert model.calls == 1
    model._audio_box_revision = 'second'
    vc.build_voice_clone_prompt(model, ref, tmp_path, False, False)
    assert model.calls == 2
    reference.write_bytes(b'changed audio')
    vc.build_voice_clone_prompt(model, ref, tmp_path, False, False)
    assert model.calls == 3
    ref.transcript = 'Changed words'
    vc.build_voice_clone_prompt(model, ref, tmp_path, False, False)
    assert model.calls == 4
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    assert len(meta['reference_sha256']) == 64
    path.with_suffix('.meta.json').write_text('{}')
    vc.build_voice_clone_prompt(model, ref, tmp_path, True, False)
    assert model.calls == 5


def test_new_legacy_prompt_full_id_and_hex_prefix_pass_cache_gate(tmp_path):
    def load_app(name, app):
        spec = importlib.util.spec_from_file_location(name, ROOT / f'apps/{app}/{app}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    vc = load_app('voice_clone_legacy_roundtrip', 'voice-clone')
    vs = load_app('voice_synth_legacy_roundtrip', 'voice-synth')
    reference = tmp_path / 'reference.wav'; reference.write_bytes(b'approved local reference')
    ref = SimpleNamespace(ref_hash='abcdef1234567890', segment=reference, transcript='Hello',
                          ref_language='English', ref_language_prob=1.0, seg_start=0.0, seg_end=1.0)
    class Model:
        name_or_path = vc.DEFAULT_TTS_MODEL
        _audio_box_checkpoint = vc.DEFAULT_TTS_MODEL
        _audio_box_revision = 'pinned-sha'
        calls = 0
        def create_voice_clone_prompt(self, **kwargs):
            self.calls += 1
            return {'local_prompt': True}
    model = Model()
    _, prompt = vc.build_voice_clone_prompt(model, ref, tmp_path, False, False)
    for identifier in (prompt.stem, ref.ref_hash[:8]):
        resolved = vs.resolve_voice(identifier, tmp_path)
        assert Path(resolved['prompt_path']) == prompt
        vs._validate_prompt_cache(vs.VoiceRegistry(tmp_path), prompt, model, model._audio_box_checkpoint)
    prompt.with_suffix('.ref.wav').unlink()
    vc.build_voice_clone_prompt(model, ref, tmp_path, False, False)
    assert model.calls == 2
    vs._validate_prompt_cache(vs.VoiceRegistry(tmp_path), prompt, model, model._audio_box_checkpoint)
