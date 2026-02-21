from pathlib import Path
import sys

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from audio import analyse_acoustics, rank_take_selection, score_take_selection  # noqa: E402


def _write_wav(path: Path, samples: np.ndarray, sr: int = 24_000) -> None:
    sf.write(str(path), samples.astype(np.float32), sr)


def test_analyse_acoustics_detects_clipping(tmp_path: Path) -> None:
    sr = 24_000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    clean = 0.15 * np.sin(2 * np.pi * 220 * t)
    clipped = np.clip(1.4 * np.sin(2 * np.pi * 220 * t), -1.0, 1.0)

    clean_path = tmp_path / "clean.wav"
    clipped_path = tmp_path / "clipped.wav"
    _write_wav(clean_path, clean, sr)
    _write_wav(clipped_path, clipped, sr)

    clean_metrics = analyse_acoustics(clean_path)
    clipped_metrics = analyse_acoustics(clipped_path)

    assert clipped_metrics["clipping_ratio"] > clean_metrics["clipping_ratio"]
    assert clipped_metrics["clipping_score"] < clean_metrics["clipping_score"]


def test_score_take_selection_returns_bounded_metrics() -> None:
    metrics = score_take_selection(
        text="Short test sentence for pacing and duration.",
        duration_sec=2.8,
        intelligibility=0.92,
        acoustics={
            "speech_ratio": 0.55,
            "speech_continuity": 0.82,
        },
    )
    assert 0.0 <= metrics["final_score"] <= 1.0
    assert 0.0 <= metrics["pacing_sanity"] <= 1.0
    assert 0.0 <= metrics["duration_fit"] <= 1.0


def test_rank_take_selection_is_deterministic_for_ties() -> None:
    takes = [
        {
            "take": "take_02",
            "selection_metrics": {
                "final_score": 0.8,
                "intelligibility": 0.8,
                "pacing_sanity": 0.8,
                "duration_fit": 0.8,
            },
        },
        {
            "take": "take_01",
            "selection_metrics": {
                "final_score": 0.8,
                "intelligibility": 0.8,
                "pacing_sanity": 0.8,
                "duration_fit": 0.8,
            },
        },
    ]
    ranked = rank_take_selection(takes)
    assert [t["take"] for t in ranked] == ["take_01", "take_02"]
    assert [t["selection_rank"] for t in ranked] == [1, 2]
