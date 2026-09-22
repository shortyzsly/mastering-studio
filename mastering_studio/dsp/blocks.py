"""Block-wise (streaming) versions of the whole-file operations.

A 20-minute mono file is 57 M samples. Doing a zero-phase filter, an FFT
convolution, a gain interpolation or a loudness measurement on the whole array
at once needs several full-length float64 temporaries: 1.3-2.9 GB *each*, which
is more than this kind of machine has free when several files are processed in
parallel. Every helper here works in ~44 s chunks with a small overlap margin
instead, and reproduces the whole-file result (see tests in the project notes:
differences are below -110 dBFS).
"""
from __future__ import annotations

import numpy as np
import pyloudnorm as pyln
from scipy import signal

CHUNK = 1 << 21          # ~44 s at 48 kHz
MARGIN_S = 1.0           # filter warm-up margin; every filter here settles in << 100 ms


def _cols(x: np.ndarray) -> np.ndarray:
    return x[:, None] if x.ndim == 1 else x


def sosfiltfilt_blocks(sos: np.ndarray, x: np.ndarray, sr: int, offset: float = 0.0, chunk: int = CHUNK) -> np.ndarray:
    """Zero-phase filter of a 1-D array in chunks. Returns float32."""
    n = len(x)
    m = int(MARGIN_S * sr)
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        a, b = max(0, s - m), min(n, e + m)
        y = signal.sosfiltfilt(sos, x[a:b].astype(np.float64) - offset)
        out[s:e] = y[s - a : s - a + (e - s)]
    return out


def fir_same_blocks(x: np.ndarray, fir: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """Linear-phase FIR, output aligned to input ('same' mode), in chunks."""
    n, L = len(x), len(fir)
    half = (L - 1) // 2
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        ext = np.zeros((e - s) + L - 1, dtype=np.float64)
        a, b = s - half, e + (L - 1 - half)
        ca, cb = max(a, 0), min(b, n)
        ext[ca - a : cb - a] = x[ca:cb]
        out[s:e] = signal.fftconvolve(ext, fir, mode="valid")[: e - s]
    return out


def block_power_db(x: np.ndarray, block: int, band_sos: np.ndarray | None = None, sr: int = 0) -> np.ndarray:
    """10*log10 of the mean square per `block` samples. Channels are linked the
    way the original whole-file code did it: per-sample max of the channels'
    squares, then the block mean. If band_sos is given, each channel is
    band-filtered first (zero-phase, chunked)."""
    cols = _cols(x)
    n = len(cols)
    nb = n // block
    chunk = (CHUNK // block) * block
    p = np.zeros(nb)
    m = int(MARGIN_S * sr)
    for s in range(0, nb * block, chunk):
        e = min(nb * block, s + chunk)
        sq = None
        for c in range(cols.shape[1]):
            if band_sos is not None:
                a, b = max(0, s - m), min(n, e + m)
                seg = signal.sosfiltfilt(band_sos, cols[a:b, c].astype(np.float64))[s - a : s - a + (e - s)]
            else:
                seg = cols[s:e, c].astype(np.float64)
            seg = seg * seg
            sq = seg if sq is None else np.maximum(sq, seg)
        p[s // block : e // block] = sq.reshape(-1, block).mean(axis=1)
    return 10 * np.log10(p + 1e-14)


def apply_gain_db(x: np.ndarray, centers: np.ndarray, gain_db: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """out = x * 10**(interp(gain_db at centers)/20), sample-rate interpolation
    done chunk by chunk. x is (n,) or (n, ch); returns float32 (n,) or (n, ch)."""
    n = len(x)
    out = np.empty(x.shape, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        g = (10 ** (np.interp(np.arange(s, e), centers, gain_db) / 20.0)).astype(np.float32)
        out[s:e] = x[s:e] * (g if x.ndim == 1 else g[:, None])
    return out


def subtract_band_with_gain(x: np.ndarray, band_sos: np.ndarray, sr: int, centers: np.ndarray, reduction_db: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """out = x - band(x) * (1 - 10**(-interp(reduction_db)/20)) with the band
    recomputed per chunk (zero-phase, with warm-up margin) so it is never stored
    for the whole file."""
    cols = _cols(x)
    n = len(cols)
    m = int(MARGIN_S * sr)
    out = np.empty(cols.shape, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        a, b = max(0, s - m), min(n, e + m)
        gain = (10 ** (-np.interp(np.arange(s, e), centers, reduction_db) / 20.0)).astype(np.float32)
        for c in range(cols.shape[1]):
            band = signal.sosfiltfilt(band_sos, cols[a:b, c].astype(np.float64))[s - a : s - a + (e - s)].astype(np.float32)
            out[s:e, c] = cols[s:e, c] - band * (1.0 - gain)
    return out[:, 0] if x.ndim == 1 else out


# --- loudness: ITU-R BS.1770 integrated LUFS, streamed ------------------------------------------------

def lufs_integrated(x: np.ndarray, sr: int) -> float:
    """Same algorithm and filter coefficients as pyloudnorm.Meter.integrated_loudness
    (K-weighting, 400 ms blocks with 75% overlap, -70 LUFS absolute and -10 LU
    relative gates), but streaming: memory is O(chunk), not O(file)."""
    step = 0.1 * sr
    if abs(step - round(step)) > 1e-9:                       # odd sample rates: use the reference implementation
        return float(pyln.Meter(sr).integrated_loudness(x.astype(np.float64)))
    step = int(round(step))
    cols = _cols(x)
    n, nch = cols.shape
    meter = pyln.Meter(sr)
    stages = [signal.tf2zpk(f.b, f.a) and (np.asarray(f.b, dtype=np.float64), np.asarray(f.a, dtype=np.float64)) for f in meter._filters.values()]
    G = np.array([1.0, 1.0, 1.0, 1.41, 1.41])[:nch]
    n_steps = -(-n // step)
    e_step = np.zeros((nch, n_steps))
    chunk = (CHUNK // step) * step
    for c in range(nch):
        zi = [np.zeros(max(len(b), len(a)) - 1) for b, a in stages]
        for s in range(0, n, chunk):
            y = cols[s : s + chunk, c].astype(np.float64)
            for i, (b, a) in enumerate(stages):
                y, zi[i] = signal.lfilter(b, a, y, zi=zi[i])
            sq = y * y
            pad = (-len(sq)) % step
            if pad:
                sq = np.concatenate([sq, np.zeros(pad)])
            e_step[c, s // step : s // step + len(sq) // step] = sq.reshape(-1, step).sum(axis=1)
    T_g = 0.4
    T = n / sr
    num_blocks = int(np.round((T - T_g) / (T_g * 0.25)) + 1)
    if num_blocks < 1:
        return -70.0
    pad = np.concatenate([e_step, np.zeros((nch, 4))], axis=1)
    z = sum(pad[:, k : k + num_blocks] for k in range(4)) / (T_g * sr)          # (nch, num_blocks)
    with np.errstate(divide="ignore"):
        l = -0.691 + 10.0 * np.log10(np.sum(G[:, None] * z, axis=0))
    idx = np.flatnonzero(l >= -70.0)
    if len(idx) == 0:
        return -70.0
    gamma_r = -0.691 + 10.0 * np.log10(np.sum(G * z[:, idx].mean(axis=1))) - 10.0
    idx = np.flatnonzero((l > gamma_r) & (l > -70.0))
    if len(idx) == 0:
        return -70.0
    with np.errstate(divide="ignore"):
        return float(-0.691 + 10.0 * np.log10(np.sum(G * z[:, idx].mean(axis=1))))
