"""Gentle speech compressor with gain-smoothing (not envelope-following).

What the first version got wrong: a fixed absolute threshold (-26 dB, 3:1)
that was not related to how loud the narrator actually is, so it was either
doing nothing or squashing everything, plus an auto-"makeup" that lifted the
whole file (room tone with it).

Here the threshold is *relative to the measured active-speech level* and the
ratio is low (1.5-2.5:1) -- levelling, not squashing. It is downward-only, so
it can never raise the noise floor: anything below threshold (all the gaps)
gets exactly 0 dB of gain change. Final level is set once, by the loudness
stage.

Detector: RMS over 10 ms blocks (closer to perceived level than peak).
Smoothing is applied to the computed gain reduction (10 ms attack, 200 ms
release), which is how hardware/plugin compressors avoid distortion.
"""
from __future__ import annotations

import numpy as np

from . import blocks

BLOCK_MS = 5.0


def auto_settings(speech_db: float, crest_db: float) -> tuple[float, float]:
    """(threshold_db, ratio) relative to the speech level.

    speech_db is active-speech RMS in dB (10*log10 mean-square).
    crest_db is the p95 frame level minus speech level; a high crest means
    peaky delivery that benefits from a touch more ratio.
    """
    ratio = 2.5 if crest_db > 9 else 2.0 if crest_db > 6 else 1.6
    return speech_db + 1.0, ratio  # only the louder passages engage


def compress(
    x: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float = 10.0,
    release_ms: float = 200.0,
    knee_db: float = 8.0,
) -> np.ndarray:
    """x is (n,) or (n, ch); channels share one gain (linked)."""
    block = max(1, int(sr * BLOCK_MS / 1000))
    n_blocks = len(x) // block
    if n_blocks < 2:
        return x
    level_db = blocks.block_power_db(x, block)  # ~10 ms detector via smoothing below
    level_db = np.convolve(level_db, np.ones(2) / 2, mode="same")

    over = level_db - threshold_db
    slope = 1.0 - 1.0 / ratio
    gr = np.zeros_like(level_db)
    upper = over >= knee_db / 2
    gr[upper] = over[upper] * slope
    knee = (over > -knee_db / 2) & ~upper
    gr[knee] = slope * (over[knee] + knee_db / 2) ** 2 / (2 * knee_db)

    a = np.exp(-BLOCK_MS / attack_ms)
    r = np.exp(-BLOCK_MS / release_ms)
    sm = np.empty_like(gr)
    prev = 0.0
    for i, g in enumerate(gr):
        prev = (a if g > prev else r) * prev + (1 - (a if g > prev else r)) * g
        sm[i] = prev

    centers = (np.arange(n_blocks) + 0.5) * block
    return blocks.apply_gain_db(x, centers, -sm)
