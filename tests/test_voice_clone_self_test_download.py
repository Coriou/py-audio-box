import importlib.util
import io
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

VOICE_CLONE_PATH = ROOT / "apps" / "voice-clone" / "voice-clone.py"


def _load_voice_clone_module():
    spec = importlib.util.spec_from_file_location("voice_clone_self_test_dl", VOICE_CLONE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def test_download_self_test_reference_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vc = _load_voice_clone_module()
    dest = tmp_path / "self_test_ref.wav"
    payload = b"RIFFxxxxWAVE"
    attempts: list[str] = []

    def _fake_urlopen(req, timeout=30):  # noqa: ANN001, ARG001
        url = req.full_url
        attempts.append(url)
        if len(attempts) == 1:
            raise RuntimeError("403 forbidden")
        return _FakeResponse(payload)

    monkeypatch.setattr(vc.urllib.request, "urlopen", _fake_urlopen)

    source = vc._download_self_test_reference(
        dest,
        sources=(
            {"url": "https://example.invalid/primary.wav", "expected_text": "primary text"},
            {"url": "https://example.invalid/fallback.wav", "expected_text": "fallback text"},
        ),
    )

    assert source["url"] == "https://example.invalid/fallback.wav"
    assert source["expected_text"] == "fallback text"
    assert attempts == [
        "https://example.invalid/primary.wav",
        "https://example.invalid/fallback.wav",
    ]
    assert dest.read_bytes() == payload


def test_download_self_test_reference_raises_when_all_sources_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vc = _load_voice_clone_module()
    dest = tmp_path / "self_test_ref.wav"

    def _always_fail(*_args, **_kwargs):  # noqa: ANN001
        raise RuntimeError("network down")

    monkeypatch.setattr(vc.urllib.request, "urlopen", _always_fail)

    with pytest.raises(RuntimeError):
        vc._download_self_test_reference(
            dest,
            sources=({"url": "https://example.invalid/primary.wav"},),
        )
    assert not dest.exists()
