"""Loudness normalization: one measured gain, then the true-peak limiter.

Targets are integrated LUFS (ITU-R BS.1770, gated), the same measurement
Auphonic, broadcast and streaming platforms use. Gain is applied ONCE and the
limiter only catches the rare peaks above the ceiling; the limiter's own
gain reduction slightly lowers the loudness, so we re-measure and trim the
gain (max 3 passes) until the *final* file lands on target.

Presets (LUFS integrated / true-peak ceiling):
  Audiobook  -23 LUFS / -3 dBTP   natural narration level (EBU R128 speech)
  ACX        -21.5 LUFS / -3 dBTP whole-file RMS lands ~-21.9 dB, inside ACX's
                                   -23..-18 window (the -23 Audiobook preset
                                   measures -23.4 RMS: 0.4 dB too quiet for ACX)
  Podcast    -16 LUFS / -1 dBTP
"""
from __future__ import annotations

import numpy as np

from . import blocks, limiter

AUDIOBOOK_LUFS = -23.0
ACX_LUFS = -21.5   # whole-file RMS ~ -21.9 dB: inside ACX's -23..-18 window with ~1.1 dB of margin (-20 LUFS was overkill, -23 was 0.4 dB under)
PODCAST_LUFS = -16.0
DEFAULT_CEILING_DBFS = -3.0
MAX_GAIN_DB = 24.0


def measure_lufs(x: np.ndarray, sr: int) -> float:
    try:
        v = blocks.lufs_integrated(x, sr)
    except Exception:
        return -70.0
    return v if np.isfinite(v) else -70.0


def gain_to_target_db(x: np.ndarray, sr: int, target_lufs: float) -> float:
    return float(np.clip(target_lufs - measure_lufs(x, sr), -MAX_GAIN_DB, MAX_GAIN_DB))


def normalize_and_limit(
    x: np.ndarray,
    sr: int,
    target_lufs: float,
    ceiling_dbfs: float,
    apply_gain: bool = True,
    apply_limit: bool = True,
) -> np.ndarray:
    if not apply_gain:
        return limiter.limit(x, sr, ceiling_dbfs) if apply_limit else x

    gain_db = gain_to_target_db(x, sr, target_lufs)
    y = x * np.float32(10 ** (gain_db / 20.0))
    if not apply_limit:
        return y.astype(np.float32)

    out = limiter.limit(y, sr, ceiling_dbfs)
    for _ in range(3):
        err = target_lufs - measure_lufs(out, sr)
        if abs(err) < 0.15:
            break
        gain_db += err
        out = limiter.limit(x * np.float32(10 ** (gain_db / 20.0)), sr, ceiling_dbfs)
    return out
