#!/usr/bin/env python3
"""Shared CLAP audio/text embedding for retrieval.

Uses ``laion/clap-htsat-fused`` through transformers (torch 2.2 CPU — the
newest torch still shipped for macOS x86_64). Audio is resampled to the
model's native 48 kHz; short clips are tile-repeated up to the 10 s training
window so transients are not diluted by zero padding.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any

import numpy as np

MODEL_ID = "laion/clap-htsat-fused"
CLAP_SR = 48_000
CLAP_WINDOW_S = 10.0


@lru_cache(maxsize=1)
def load_clap() -> dict[str, Any]:
    """Load and memoize the model bundle (downloads on first use)."""
    import torch
    from transformers import ClapModel, ClapProcessor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ClapModel.from_pretrained(MODEL_ID)
        processor = ClapProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return {"model": model, "processor": processor, "torch": torch}


def _tile_to_window(audio: np.ndarray) -> np.ndarray:
    """Repeat short audio up to (crop to) one CLAP training window."""
    target = int(CLAP_WINDOW_S * CLAP_SR)
    if audio.size == 0:
        return np.zeros(target, dtype=np.float32)
    if audio.size >= target:
        return audio[:target]
    repeats = int(np.ceil(target / audio.size))
    return np.tile(audio, repeats)[:target]


@lru_cache(maxsize=1)
def label_embeddings(texts: tuple[str, ...]) -> np.ndarray:
    """Normalized CLAP text embeddings for a fixed label set (row per text)."""
    bundle = load_clap()
    torch = bundle["torch"]
    model, processor = bundle["model"], bundle["processor"]
    with torch.no_grad():
        inputs = processor(text=list(texts), return_tensors="pt", padding=True)
        features = model.get_text_features(**inputs)
    features = torch.nn.functional.normalize(features, dim=-1)
    return features.numpy()


def embed_audio(audio_48k: np.ndarray) -> np.ndarray:
    """Normalized CLAP audio embedding for mono float PCM at 48 kHz."""
    bundle = load_clap()
    torch = bundle["torch"]
    model, processor = bundle["model"], bundle["processor"]
    windowed = _tile_to_window(np.asarray(audio_48k, dtype=np.float32))
    with torch.no_grad():
        inputs = processor(
            audio=windowed, sampling_rate=CLAP_SR, return_tensors="pt",
        )
        features = model.get_audio_features(**inputs)
    features = torch.nn.functional.normalize(features, dim=-1)
    return features.numpy()[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
