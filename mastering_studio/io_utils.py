from __future__ import annotations

import os

import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3"}


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Returns (data, sample_rate). data is shape (n_samples,) for mono or
    (n_samples, n_channels) for multi-channel, float32 in [-1, 1]."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    return data, sr


def to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return np.mean(data, axis=1).astype(np.float32)


def save_audio(path: str, data: np.ndarray, sr: int, subtype: str | None = None) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if subtype is None:
        subtype = "PCM_24" if path.lower().endswith((".wav", ".aiff", ".aif")) else None
    sf.write(path, data, sr, subtype=subtype)
