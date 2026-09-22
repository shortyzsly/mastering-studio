"""Mouth-click removal: detect isolated 1-3 ms broadband spikes, repair by
autoregressive (LPC) interpolation.

What the first version got wrong: it flagged any sample whose second
derivative was a statistical outlier -- which is most of speech (~18
"clicks" per second on a real chapter) -- and replaced them with smoothstep
ramps, shaving 3 dB off speech peaks. Real tools (iZotope RX Mouth De-click,
Audacity's click removal) instead look for events that are *short and
isolated* relative to their surroundings, then rebuild the gap from the
signal's own spectral model.

A click here must satisfy all of:
  * high-frequency (>3 kHz) energy in a 1 ms block at least 10x (20 dB) the
    *median* level of the surrounding ~40 ms (median, so speech and the click
    itself don't inflate the baseline);
  * neighbouring blocks are back near baseline (event is isolated, <= ~3 ms;
    plosive bursts and fricatives last 10-100 ms and are left alone).
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz
from scipy.ndimage import median_filter

from . import blocks as blocks_mod

BLOCK_MS = 1.0
CONTEXT = 384          # samples of context per side for the AR fit
AR_ORDER = 24
SPIKE_RATIO = 10.0     # peak / local median level
NEIGHBOR_RATIO = 4.0   # neighbours must be below this (isolated)
ABS_FLOOR = 10 ** (-62 / 20)  # ignore spikes below -62 dBFS (inaudible)


def _lpc(seg: np.ndarray, order: int) -> np.ndarray | None:
    seg = np.asarray(seg, dtype=np.float64)
    seg = seg - seg.mean()
    r = np.correlate(seg, seg, mode="full")[len(seg) - 1 : len(seg) + order]
    if r[0] <= 1e-12:
        return None
    r[0] *= 1.0 + 1e-6
    try:
        return solve_toeplitz(r[:order], r[1 : order + 1])
    except Exception:
        return None


def _predict(history: np.ndarray, a: np.ndarray, n: int) -> np.ndarray:
    order = len(a)
    buf = list(history[-order:])
    out = np.empty(n)
    for i in range(n):
        out[i] = np.dot(a, buf[::-1][:order])
        buf.append(out[i])
        buf.pop(0)
    return out


def _repair(x: np.ndarray, s: int, e: int) -> bool:
    n = e - s
    if s < CONTEXT or e + CONTEXT >= len(x):
        return False
    left, right = np.asarray(x[s - CONTEXT : s], dtype=np.float64), np.asarray(x[e : e + CONTEXT], dtype=np.float64)
    a_l, a_r = _lpc(left, AR_ORDER), _lpc(right[::-1], AR_ORDER)
    if a_l is None or a_r is None:
        return False
    fwd = _predict(left, a_l, n)
    bwd = _predict(right[::-1], a_r, n)[::-1]
    w = np.linspace(1.0, 0.0, n)
    fill = w * fwd + (1.0 - w) * bwd
    # Guard: a repair must not be louder than its own surroundings.
    ctx_rms = np.sqrt(np.mean(np.concatenate([left[-64:], right[:64]]) ** 2)) + 1e-9
    if not np.all(np.isfinite(fill)) or np.sqrt(np.mean(fill ** 2)) > 2.0 * ctx_rms:
        return False
    x[s:e] = fill
    return True


def find_clicks(x: np.ndarray, sr: int) -> list[tuple[int, int]]:
    block = max(8, int(sr * BLOCK_MS / 1000))
    n_blocks = len(x) // block
    if n_blocks < 64:
        return []
    sos = signal.butter(4, 3000, btype="highpass", fs=sr, output="sos")
    env = np.empty(n_blocks)
    zi = np.zeros((sos.shape[0], 2))
    chunk = (blocks_mod.CHUNK // block) * block
    for s0 in range(0, n_blocks * block, chunk):
        e0 = min(n_blocks * block, s0 + chunk)
        y, zi = signal.sosfilt(sos, np.asarray(x[s0:e0], dtype=np.float64), zi=zi)
        env[s0 // block : e0 // block] = np.abs(y).reshape(-1, block).max(axis=1)
    base = median_filter(env, size=41, mode="nearest") + 1e-9  # ~40 ms

    spike = (env > SPIKE_RATIO * base) & (env > ABS_FLOOR)
    clicks = []
    for b in np.flatnonzero(spike):
        lo, hi = max(0, b - 3), min(n_blocks, b + 4)
        # isolated: blocks 2-3 away on both sides are back near baseline
        ring = np.r_[env[lo : max(lo, b - 1)], env[min(hi, b + 2) : hi]]
        if len(ring) and np.any(ring > NEIGHBOR_RATIO * base[b]):
            continue
        clicks.append((max(0, (b - 1) * block), min(len(x), (b + 2) * block)))
    # merge overlapping spans
    merged: list[list[int]] = []
    for s, e in clicks:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if (e - s) <= int(sr * 0.004)]


def declick(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Returns (audio, number of clicks repaired)."""
    spans = find_clicks(x, sr)
    if not spans:
        return x, 0
    y = np.array(x, dtype=np.float32, copy=True)
    fixed = sum(_repair(y, s, e) for s, e in spans)
    return y, int(fixed)
