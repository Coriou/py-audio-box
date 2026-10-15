"""Public clone helper must reject cross-checkpoint cache reuse before inference."""
import pickle
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import tts
from tts_backends.contract import BackendError


SMALL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
LARGE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def model(checkpoint=SMALL, revision="commit-a"):
    return SimpleNamespace(name_or_path=checkpoint, _audio_box_checkpoint=checkpoint,
                           _audio_box_revision=revision)


def forbid_generation(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("incompatible prompt reached generation")
    monkeypatch.setattr(tts, "_generate", fail)


def test_external_script_cached_small_prompt_rejected_by_large_model(monkeypatch):
    # Simulates a custom script loading its own trusted local cache and calling
    # the public helper, bypassing voice-synth's CLI cache gate entirely.
    cached = pickle.dumps(tts.bind_clone_prompt([{"ref_text": "Approved words"}], model()))
    prompt = pickle.loads(cached)
    forbid_generation(monkeypatch)
    with pytest.raises(BackendError, match="authorized reference") as error:
        tts.synthesise_clone("Hello", "English", model(LARGE), prompt, 1)
    assert error.value.code == "clone_prompt_mismatch"


@pytest.mark.parametrize("revision", ["commit-b", None])
def test_revision_mismatch_and_unknown_revision_rejected(monkeypatch, revision):
    forbid_generation(monkeypatch)
    prompt = tts.bind_clone_prompt([object()], model())
    with pytest.raises(BackendError) as error:
        tts.synthesise_clone("Hello", "English", model(revision=revision), prompt, None)
    assert error.value.code == "clone_prompt_mismatch"


def test_trusted_legacy_payload_metadata_passes_and_unwraps(monkeypatch):
    payload = [{"ref_text": "Original transcript", "icl_mode": True}]
    captured = {}
    def generate(target, method, params, *args):
        captured.update(params)
        return "audio", 24000
    monkeypatch.setattr(tts, "_generate", generate)
    result = tts.synthesise_clone("Hello", "English", model(), payload, None,
                                 prompt_metadata={"checkpoint": SMALL, "revision": "commit-a"})
    assert result == ("audio", 24000)
    assert captured["voice_clone_prompt"] is payload


def test_unproven_raw_prompt_rejected_before_generation(monkeypatch):
    forbid_generation(monkeypatch)
    with pytest.raises(BackendError) as error:
        tts.synthesise_clone("Hello", "English", model(), [object()], None)
    assert error.value.code == "clone_prompt_identity_missing"


def test_bound_prompt_cannot_be_relabelled_with_target_metadata():
    prompt = tts.bind_clone_prompt([object()], model())
    with pytest.raises(BackendError):
        tts.bind_clone_prompt(prompt, metadata={"checkpoint": LARGE, "revision": "commit-a"})
    with pytest.raises(BackendError):
        tts.bind_clone_prompt(prompt, model(LARGE))


def test_lazy_backend_resolves_requested_tag_to_actual_revision(monkeypatch):
    class LazyBackend:
        name_or_path = _audio_box_checkpoint = SMALL
        _audio_box_revision = "main"
        resolutions = 0
        def get_supported_languages(self):
            self.resolutions += 1
            self._audio_box_revision = "commit-a"
            return ["English"]
    monkeypatch.setattr(tts, "QwenBackend", LazyBackend)
    target = LazyBackend()
    prompt = tts.bind_clone_prompt([object()], model())
    assert tts.validate_clone_prompt(prompt, target) is prompt.payload
    assert target.resolutions == 1


def test_trusted_cache_metadata_must_agree_with_model():
    with pytest.raises(BackendError):
        tts.bind_clone_prompt([object()], model(LARGE), metadata={"checkpoint": SMALL, "revision": "commit-a"})


def test_mutable_local_checkpoint_alias_refused_even_with_revision_label(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    alias = tmp_path / "approved-alias"
    alias.symlink_to(checkpoint, target_is_directory=True)
    payload = [object()]
    prompt = tts.bind_clone_prompt(payload, model(str(checkpoint)))
    with pytest.raises(BackendError, match="Mutable local"):
        tts.validate_clone_prompt(prompt, model(str(alias)))


@pytest.mark.parametrize("revision", [None, ""])
def test_both_prompt_and_model_missing_revisions_refused(monkeypatch, revision):
    forbid_generation(monkeypatch)
    prompt = tts.bind_clone_prompt([object()], model(revision=revision))
    with pytest.raises(BackendError, match="revision is unresolved") as error:
        tts.synthesise_clone("Hello", "English", model(revision=revision), prompt, None)
    assert error.value.code == "clone_prompt_identity_missing"


@pytest.mark.parametrize("revision", [None, "unverified-local-label"])
def test_same_local_path_checkpoint_replacement_cannot_reuse_prompt(tmp_path, monkeypatch, revision):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    config.write_text(json.dumps({"model_type": "qwen3_tts", "tts_model_size": "0b6", "tts_model_type": "base"}))
    prompt = tts.bind_clone_prompt([object()], model(str(checkpoint), revision))
    config.write_text(json.dumps({"model_type": "qwen3_tts", "tts_model_size": "1b7", "tts_model_type": "base"}))
    forbid_generation(monkeypatch)
    with pytest.raises(BackendError) as error:
        tts.synthesise_clone("Hello", "English", model(str(checkpoint), revision), prompt, None)
    assert error.value.code == "clone_prompt_identity_missing"


def test_snapshot_local_alias_matches_repo_and_immutable_revision(tmp_path):
    snapshot = tmp_path / "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base" / "snapshots" / "commit-a"
    snapshot.mkdir(parents=True)
    alias = tmp_path / "local-base"
    alias.symlink_to(snapshot, target_is_directory=True)
    prompt = tts.bind_clone_prompt([object()], model(str(alias), None))
    assert prompt.checkpoint == SMALL
    assert prompt.revision == "commit-a"
    assert tts.validate_clone_prompt(prompt, model()) is prompt.payload
    with pytest.raises(BackendError):
        tts.bind_clone_prompt([object()], model(str(alias), "commit-b"))


def test_same_basename_different_local_checkpoints_not_equated(tmp_path):
    a, b = tmp_path / "a" / "Base", tmp_path / "b" / "Base"
    a.mkdir(parents=True); b.mkdir(parents=True)
    prompt = tts.bind_clone_prompt([object()], model(str(a)))
    with pytest.raises(BackendError):
        tts.validate_clone_prompt(prompt, model(str(b)))


def test_generation_kwargs_cannot_override_validated_prompt(monkeypatch):
    payload = [object()]
    prompt = tts.bind_clone_prompt(payload, model())
    def generate(target, method, params, *args):
        assert params["voice_clone_prompt"] is payload
        return "audio", 24000
    monkeypatch.setattr(tts, "_generate", generate)
    tts.synthesise_clone("Hello", "English", model(), prompt, None,
                         gen_kwargs={"voice_clone_prompt": ["foreign prompt"]})
