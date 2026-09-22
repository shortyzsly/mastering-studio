"""True-peak lookahead brickwall limiter (4x oversampled detection).

Same design as broadcast/mastering limiters:
  1. Detect peaks on a 4x-oversampled signal, so inter-sample peaks that a
     DAC/codec would reconstruct above the ceiling are caught (this is what
     "true peak" / dBTP means, and what Auphonic's limiter does).
  2. Required gain per sample -> sliding *minimum* over the lookahead window
     -> sliding *mean* over half that window. That guarantees the gain is
     already down before the peak arrives and ramps smoothly (no distortion
     from abrupt gain steps), while never exceeding what any sample needs.
  3. Fast attack (the lookahead), program-dependent release (exponential,
     80 ms) so the level recovers without pumping.

Channels are linked (one gain for all) so the stereo image doesn't shift.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

OVERSAMPLE = 4
CHUNK = 1 << 20
PAD = 64


def true_peak_envelope(x: np.ndarray) -> np.ndarray:
    """Per-sample max |value| over the 4x-oversampled neighbourhood
    (channels combined by max). Chunked so memory stays small."""
    chans = x[:, None] if x.ndim == 1 else x
    n = len(chans)
    env = np.zeros(n, dtype=np.float32)
    for c in range(chans.shape[1]):
        for s in range(0, n, CHUNK):
            e = min(n, s + CHUNK)
            a, b = max(0, s - PAD), min(n, e + PAD)
            seg = chans[a:b, c]
            up = signal.resample_poly(seg, OVERSAMPLE, 1).astype(np.float32)
            m = np.abs(up[: (b - a) * OVERSAMPLE]).reshape(-1, OVERSAMPLE).max(axis=1)
            core = np.maximum(m[s - a : s - a + (e - s)], np.abs(seg[s - a : s - a + (e - s)]))
            env[s:e] = np.maximum(env[s:e], core)
    return env


def limit(
    x: np.ndarray,
    sr: int,
    ceiling_dbfs: float = -3.0,
    lookahead_ms: float = 3.0,
    release_ms: float = 80.0,
) -> np.ndarray:
    """Processed in ~44 s chunks (with a filter-support margin), carrying the
    release state across chunk boundaries, so memory does not grow with file
    length. Each sample's gain is still min()'d with the value that guarantees it
    is under the ceiling, so peak safety does not depend on the chunking."""
    ceiling = 10 ** (ceiling_dbfs / 20.0)
    env = true_peak_envelope(x)
    if env.max() <= ceiling:
        return x

    n = len(env)
    look = max(2, int(sr * lookahead_ms / 1000.0))
    block = max(1, int(sr * 0.001))
    coef = np.exp(-1.0 / ((sr / block) * release_ms / 1000.0))
    margin = 2 * look + 2
    chunk = (CHUNK // block) * block
    out = np.empty(x.shape, dtype=np.float32)
    prev = 1.0
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        a, b = max(0, s - margin), min(n, e + margin)
        g_req = np.minimum(1.0, ceiling / np.maximum(env[a:b], 1e-9)).astype(np.float32)
        g_min = minimum_filter1d(g_req, size=2 * look + 1, mode="nearest")
        g_att = uniform_filter1d(g_min, size=look + 1, mode="nearest")
        core = g_att[s - a : s - a + (e - s)]

        # Release: block-rate exponential recovery, never above g_att.
        nbk = -(-(e - s) // block)
        padded = np.pad(core, (0, nbk * block - (e - s)), constant_values=1.0)
        g_blk = padded.reshape(nbk, block).min(axis=1)
        rel = np.empty(nbk, dtype=np.float32)
        for i, g in enumerate(g_blk):
            prev = g if g < prev else coef * prev + (1 - coef) * g
            rel[i] = prev
        centers = (np.arange(nbk) + 0.5) * block
        gain = np.minimum(np.interp(np.arange(e - s), centers, rel).astype(np.float32), core)
        out[s:e] = x[s:e] * (gain if x.ndim == 1 else gain[:, None])
    return out
