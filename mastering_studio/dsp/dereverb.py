"""De-reverb for small-booth reflections: blind decay-time (T60) estimate, then
statistical late-reverberation suppression (Lebart et al. 2001 / Habets 2007).

Model: after the direct sound and early reflections, a room's response decays
exponentially, so the reverberant power at time t is approximately the signal
power Td seconds earlier times exp(-2*delta*Td), delta = 3 ln10 / T60. Where
the signal is falling (word tails, the "boxy ring" after each phrase) that
estimate approaches the current power and the gain drops; at onsets and in
steady speech it is far below and the gain stays ~1. Gaps (room tone) are
stationary, so they are untouched too.

T60 is estimated from the speech itself: a room can't make a sound decay
FASTER than its own decay rate, so the steepest decays observed after word
offsets (a low percentile of the measured slopes) approximate the room's.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from . import blocks

STRENGTH_FLOOR_DB = {"light": 6.0, "medium": 10.0, "strong": 15.0}   # max suppression per bin
TD_S = 0.05            # early/late boundary: leave direct sound + first 50 ms of reflections alone
PSD_SMOOTH = 0.6       # recursive smoothing of the power spectrum (limits musical noise)
ENV_BLOCK_S = 0.005
SKIP_DB = 10.0         # fit the decay only after it has fallen this far: past the direct-sound drop
MIN_FIT_DB = 8.0
SLOPE_PERCENTILE = 35  # calibrated on synthetic rooms (T60 0.2-0.7 s, DRR 3-10 dB): within ~10%


def estimate_t60(x: np.ndarray, sr: int) -> float | None:
    """Decay time in seconds, or None if the file has too few clean word offsets."""
    block = max(1, int(ENV_BLOCK_S * sr))
    sos = signal.butter(2, [300.0, min(4000.0, 0.45 * sr)], btype="bandpass", fs=sr, output="sos")
    env = blocks.block_power_db(x, block, band_sos=sos, sr=sr)
    env = 10 * np.log10(uniform_filter1d(10 ** (env / 10), 3) + 1e-14)
    floor = float(np.percentile(env, 5))
    top = floor + 25.0
    slopes = []
    i, n = 1, len(env)
    while i < n - 1:
        if env[i] >= top and env[i] >= env[i - 1] and env[i] > env[i + 1]:
            j = i
            while j + 1 < n and env[j + 1] <= env[j] + 1.0 and env[j + 1] > floor + 6.0:
                j += 1
            seg = env[i : j + 1]
            if seg[0] - seg[-1] >= SKIP_DB + MIN_FIT_DB:
                part = seg[int(np.argmax(seg <= seg[0] - SKIP_DB)) :]
                if len(part) >= 6:
                    slopes.append(np.polyfit(np.arange(len(part)) * ENV_BLOCK_S, part, 1)[0])
            i = j + 1
        else:
            i += 1
    if len(slopes) < 10:
        return None
    rate = float(np.percentile(slopes, SLOPE_PERCENTILE))   # dB/s, steepest decays
    return float(np.clip(-60.0 / min(rate, -1.0), 0.05, 2.0))


def auto_strength(t60: float | None) -> str | None:
    if t60 is None or t60 < 0.15:
        return None
    return "light" if t60 < 0.3 else "medium" if t60 < 0.5 else "strong"


def dereverb(x: np.ndarray, sr: int, t60: float, strength: str = "medium", stats: dict | None = None) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n_fft = 1 << int(round(np.log2(0.023 * sr)))   # ~23 ms
    hop = n_fft // 4
    win = np.sqrt(np.hanning(n_fft + 1)[:-1]).astype(np.float32)
    nd = max(1, int(round(TD_S * sr / hop)))
    decay = np.exp(-2.0 * (3.0 * np.log(10.0) / t60) * nd * hop / sr)
    g_min = 10 ** (-STRENGTH_FLOOR_DB.get(strength, 10.0) / 20.0)

    n = len(x)
    n_frames = (n + n_fft) // hop + 1
    xp = np.zeros((n_frames - 1) * hop + n_fft, dtype=np.float32)
    xp[n_fft // 2 : n_fft // 2 + n] = x
    out = np.zeros_like(xp)
    zi = np.zeros((1, n_fft // 2 + 1))
    hist = np.zeros((nd, n_fft // 2 + 1))   # smoothed PSD of the last nd frames (carried across batches)
    sum_in = sum_out = 0.0
    batch = 1024
    for f0 in range(0, n_frames, batch):
        f1 = min(n_frames, f0 + batch)
        idx = (np.arange(f0, f1)[:, None] * hop) + np.arange(n_fft)[None, :]
        spec = np.fft.rfft(xp[idx] * win, axis=1)
        p = spec.real ** 2 + spec.imag ** 2
        ps, zi = signal.lfilter([1.0 - PSD_SMOOTH], [1.0, -PSD_SMOOTH], p, axis=0, zi=zi)
        both = np.concatenate([hist, ps], axis=0)
        past, hist = both[: len(ps)], both[-nd:]
        g = np.clip(1.0 - decay * past / np.maximum(ps, 1e-20), g_min, 1.0)
        g[:, 1:-1] = (g[:, :-2] + g[:, 1:-1] + g[:, 2:]) / 3.0   # 3-bin smoothing against musical noise
        sum_in += float(p.sum())
        sum_out += float((p * g * g).sum())
        frames = np.fft.irfft(spec * g, n=n_fft, axis=1).astype(np.float32) * win
        for i in range(f1 - f0):
            s = (f0 + i) * hop
            out[s : s + n_fft] += frames[i]
    if stats is not None:
        stats["dereverb_removed_db"] = round(10 * np.log10(max(sum_in, 1e-20) / max(sum_out, 1e-20)), 2)
    out *= 0.5   # sqrt-Hann^2 at 75% overlap sums to 2
    return out[n_fft // 2 : n_fft // 2 + n]
