"""Speech / room-tone activity detection shared by every stage.

Professional restoration tools (iZotope RX "Learn", Auphonic's classic
denoiser) work from the same idea: find the passages that are *only* room
tone, learn the noise from those, and measure/level speech only where speech
actually is. Estimating noise from the whole clip (what the first version
did) mixes speech into the "noise" model, so the denoiser either does
nothing or eats the voice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_MS = 20.0
HOP_MS = 10.0


@dataclass
class Activity:
    hop: int                 # samples per hop
    frame: int               # samples per frame
    level_db: np.ndarray     # per-frame power in dB (10*log10 of mean square)
    floor_db: float          # room-tone level (dB, same scale as level_db)
    speech_db: float         # active-speech RMS level (dB)
    noise_mask: np.ndarray   # frames usable as a noise profile (clean room tone)
    speech_mask: np.ndarray  # frames that contain speech


def frame_power_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, int, int]:
    frame = int(round(sr * FRAME_MS / 1000.0))
    hop = int(round(sr * HOP_MS / 1000.0))
    n_blocks = len(x) // hop
    if n_blocks < 2:
        return np.array([-120.0]), hop, frame
    blocks = x[: n_blocks * hop].reshape(n_blocks, hop)
    p = np.einsum("ij,ij->i", blocks, blocks, dtype=np.float64) / hop
    p_frame = 0.5 * (p[:-1] + p[1:])
    return 10.0 * np.log10(p_frame + 1e-12), hop, frame


def _runs(mask: np.ndarray, min_len: int, trim: int) -> np.ndarray:
    """Keep only runs of True at least min_len long, trimmed by `trim` frames
    on each side (speech decay / reverb tails sit right at run edges)."""
    out = np.zeros_like(mask)
    if not mask.any():
        return out
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= min_len:
            out[a + trim : b - trim] = True
    return out


def analyze_activity(x: np.ndarray, sr: int) -> Activity:
    level_db, hop, frame = frame_power_db(x, sr)
    floor_db = float(np.percentile(level_db, 5))
    p95 = float(np.percentile(level_db, 95))

    # Speech threshold: partway between the room tone and the loud passages,
    # but never closer than 15 dB to the floor.
    speech_thresh = floor_db + max(15.0, 0.4 * (p95 - floor_db))
    speech_mask = level_db >= speech_thresh
    if speech_mask.any():
        speech_db = float(10 * np.log10(np.mean(10 ** (level_db[speech_mask] / 10.0)) + 1e-12))
    else:
        speech_db = p95

    # Clean room tone: within 6 dB of the floor, in runs >= 250 ms, with 50 ms
    # trimmed off each edge.
    noise_mask = _runs(level_db <= floor_db + 6.0, min_len=25, trim=5)
    if noise_mask.sum() < 30:  # <300 ms found: relax, fall back to quietest 5%
        noise_mask = level_db <= np.percentile(level_db, 5)
    return Activity(hop, frame, level_db, floor_db, speech_db, noise_mask, speech_mask)


def frame_ranges(mask: np.ndarray, hop: int, frame: int, n_samples: int | None = None) -> list[tuple[int, int]]:
    """Sample ranges [start, end) covered by contiguous runs of True frames,
    clamped to n_samples (pass len(x) so callers never slice past the end)."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    cap = n_samples if n_samples is not None else 1 << 62
    return [(int(a * hop), int(min(b * hop + frame, cap))) for a, b in zip(edges[::2], edges[1::2])]
