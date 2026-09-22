"""De-hum: find mains hum/buzz (a comb of lines at multiples of 50 or 60 Hz)
in the room tone and subtract it inside the gaps.

Why a separate stage: the broadband denoiser (dsp.denoise) works on ~23 Hz
STFT bins with a smoothed noise profile, so a comb whose lines are 50 Hz apart
and a few Hz wide is blurred into a flat floor. It then lowers the tonal lines
by exactly as much as the surrounding noise, and they stay 10-19 dB above the
floor -- and tones are far more audible than the same energy as broadband
hiss.

Detection is from room tone only (dsp.activity), at 0.7 Hz resolution:
  1. average the room-tone power spectrum (65536-pt FFT);
  2. prominence of each bin = its level minus the running median of +/-40 Hz;
  3. search f0 in 50 +/- 1 Hz and 60 +/- 1 Hz (0.01 Hz steps) for the value
     whose harmonics have the greatest total prominence;
  4. accept only if >= 5 harmonics stand >= 6 dB above the bed. Otherwise the
     file has no hum and nothing is touched.

Removal is a LOCAL SINUSOIDAL MODEL, not a notch filter. Inside each gap
(frame level <= floor + 22 dB, >= 150 ms) the amplitude and phase of every
detected line is least-squares fitted to that gap's own samples and the fitted
sinusoids are subtracted, faded in/out over 30 ms at the gap edges.

Why not notch filters (the first version): a few-Hz-wide notch has an impulse
response that rings for 100+ ms, and run zero-phase it rings BEFORE loud
sounds as well. Loud speech excited 47 little tone bursts in the gaps next to
words (+6-7 dB at 80 Hz-1 kHz just outside speech). Subtracting fitted
sinusoids has no filter memory, so there is no ringing, and outside the gaps
the audio is bit-identical -- speech is never touched. Under speech the hum
sits ~50 dB below the voice and is masked.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from . import activity as act_mod

MAINS_HZ = (50.0, 60.0)
MIN_LINES = 5
MIN_PROMINENCE_DB = 6.0
MAX_HZ = 4000.0


def _room_tone_spectrum(x: np.ndarray, sr: int, act: act_mod.Activity, N: int, part: slice | None = None):
    win = np.hanning(N).astype(np.float32)
    acc, n = 0.0, 0
    ranges = act_mod.frame_ranges(act.noise_mask, act.hop, act.frame, len(x))
    if part is not None:
        ranges = ranges[part]
    for s, e in ranges:
        for i in range(s, e - N + 1, N // 2):
            acc = acc + np.abs(np.fft.rfft(x[i : i + N] * win)) ** 2
            n += 1
            if n >= 400:
                break
    if n == 0:
        return None, 0
    return 10 * np.log10(acc / n + 1e-20), n


def _prominence(P: np.ndarray, res: float) -> np.ndarray:
    size = int(80.0 / res) | 1
    return P - median_filter(P, size=size, mode="nearest")


def detect(x: np.ndarray, sr: int, act: act_mod.Activity | None = None) -> dict | None:
    act = act or act_mod.analyze_activity(x, sr)
    # Prefer the finest resolution the room tone allows; fall back to shorter
    # windows (still < 3 Hz, far finer than the 50 Hz line spacing) for files
    # with few or short pauses. Shorter windows need more averaged frames.
    P = None
    for N, min_frames in ((65536, 3), (32768, 4), (16384, 8)):
        P, n = _room_tone_spectrum(x, sr, act, N)
        if P is not None and n >= min_frames:
            break
        P = None
    if P is None:
        return None
    res = sr / N
    f = np.fft.rfftfreq(N, 1 / sr)
    prom = _prominence(P, res)

    best = (0.0, None)
    for base in MAINS_HZ:
        for f0 in np.arange(base - 1.0, base + 1.0 + 1e-9, 0.01):
            score = 0.0
            for k in range(2, int(MAX_HZ / f0) + 1):
                w = 0.5 + 0.03 * k
                m = (f >= k * f0 - w) & (f <= k * f0 + w)
                if m.any():
                    score += max(0.0, float(prom[m].max()) - 3.0)
            if score > best[0]:
                best = (score, f0)
    if best[1] is None:
        return None
    f0 = best[1]

    lines = []
    for k in range(1, int(MAX_HZ / f0) + 1):
        w = 0.5 + 0.03 * k
        m = np.flatnonzero((f >= k * f0 - w) & (f <= k * f0 + w))
        if len(m) == 0 or f[m[0]] < 40:
            continue
        j = int(m[np.argmax(prom[m])])
        if prom[j] >= MIN_PROMINENCE_DB:
            # parabolic refinement of the true line frequency
            a, b, c = P[j - 1], P[j], P[j + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
            lines.append({"k": k, "hz": float((j + d) * res), "prom_db": float(prom[j])})
    if len(lines) < MIN_LINES:
        return None

    # drift: does the strongest line move between early and late room tone?
    drift = 0.0
    ranges = act_mod.frame_ranges(act.noise_mask, act.hop, act.frame, len(x))
    if len(ranges) >= 2:
        top = sorted(lines, key=lambda l: -l["prom_db"])[:4]
        mid = len(ranges) // 2
        shifts = []
        for part in (slice(0, mid), slice(mid, None)):
            Pp, npart = _room_tone_spectrum(x, sr, act, N, part)
            if Pp is None:
                shifts = []
                break
            row = []
            for l in top:
                m = (f >= l["hz"] - 2.5) & (f <= l["hz"] + 2.5)
                row.append(f[m][np.argmax(Pp[m])])
            shifts.append(row)
        if len(shifts) == 2:
            drift = float(np.max(np.abs(np.array(shifts[0]) - np.array(shifts[1]))))
    return {"f0": float(f0), "lines": lines, "drift_hz": drift, "score": best[0], "room_tone_frames": n}


GAP_ABOVE_FLOOR_DB = 22.0
MIN_GAP_FRAMES = 15        # 150 ms
TRIM_FRAMES = 2            # keep 20 ms away from the speech at each edge
EDGE_FADE_S = 0.030
BLOCK_S = 0.6
SINGLE_BLOCK_MAX_S = 0.8


def gap_runs(act: act_mod.Activity, n_samples: int) -> list[tuple[int, int]]:
    quiet = act.level_db <= act.floor_db + GAP_ABOVE_FLOOR_DB
    padded = np.concatenate(([False], quiet, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= MIN_GAP_FRAMES:
            s = (a + TRIM_FRAMES) * act.hop
            e = min((b - TRIM_FRAMES) * act.hop + act.frame, n_samples)
            if e - s > act.hop * 8:
                out.append((int(s), int(e)))
    return out


_PHASOR_CACHE: dict = {}


def _phasors(n: int, freqs: np.ndarray, sr: int) -> np.ndarray:
    """exp(-2*pi*j*f*t) for t = 0..n-1 samples: one master table per (sr, freqs),
    sliced (as a view) for each gap, since every gap's time base starts at 0."""
    key = (sr, freqs.tobytes())
    tab = _PHASOR_CACHE.get(key)
    n_max = int(max(SINGLE_BLOCK_MAX_S, BLOCK_S) * sr) + 8
    if tab is None or len(tab) < n:
        _PHASOR_CACHE.clear()
        t = np.arange(max(n, n_max)) / sr
        tab = np.exp(-2j * np.pi * t[:, None] * freqs[None, :])
        _PHASOR_CACHE[key] = tab
    return tab[:n]


def _fit_hum(seg: np.ndarray, freqs: np.ndarray, sr: int) -> np.ndarray:
    """Amplitude+phase of a sinusoid at each of `freqs`, estimated with a
    Hann-weighted matched filter, returned as the summed synthetic hum.

    A joint least-squares fit is unnecessary: lines are >= 50 Hz apart and the
    window is >= 0.1 s, so each line's neighbours sit >= 5 bins away where the
    Hann window's response is < -60 dB. Independent estimates are therefore
    equivalent to the joint fit and ~100x cheaper."""
    n = len(seg)
    ph = _phasors(n, freqs, sr)
    w = np.hanning(n + 2)[1:-1]
    c = 2.0 * (w * seg) @ ph / w.sum()          # complex amplitude of each line (conj convention)
    return np.real(np.conj(ph) @ c)


def _subtraction_for_gap(x: np.ndarray, s: int, e: int, freqs: np.ndarray, sr: int) -> np.ndarray:
    n = e - s
    seg = x[s:e].astype(np.float64)
    if n <= SINGLE_BLOCK_MAX_S * sr:
        sub = _fit_hum(seg, freqs, sr)
    else:
        blk = int(BLOCK_S * sr)
        hop = blk // 2
        sub = np.zeros(n)
        wsum = np.zeros(n)
        starts = list(range(0, n - blk + 1, hop))
        if starts[-1] + blk < n:
            starts.append(n - blk)
        for j, b0 in enumerate(starts):
            est = _fit_hum(seg[b0 : b0 + blk], freqs, sr)
            w = np.hanning(blk + 2)[1:-1].copy()
            if j == 0:
                w[: blk // 2] = 1.0            # no fade-in inside the first block: the gap-edge fade does it
            if j == len(starts) - 1:
                w[blk // 2 :] = 1.0
            sub[b0 : b0 + blk] += w * est
            wsum[b0 : b0 + blk] += w
        sub /= np.maximum(wsum, 1e-9)
    f = max(2, int(EDGE_FADE_S * sr))
    f = min(f, n // 2)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(f) / f))
    fade = np.ones(n)
    fade[:f] = ramp
    fade[-f:] = ramp[::-1]
    return sub * fade


def dehum(x: np.ndarray, sr: int, act: act_mod.Activity | None = None, info: dict | None = None) -> tuple[np.ndarray, dict]:
    """Returns (audio, stats). stats is empty if no hum was found."""
    act = act or act_mod.analyze_activity(x, sr)
    info = info if info is not None else detect(x, sr, act)
    if not info:
        return x, {}
    freqs = np.array([l["hz"] for l in info["lines"]])
    y = x.astype(np.float32, copy=True)
    covered = 0
    for s, e in gap_runs(act, len(x)):
        y[s:e] = (x[s:e] - _subtraction_for_gap(x, s, e, freqs, sr)).astype(np.float32)
        covered += e - s
    stats = {
        "hum_f0_hz": round(info["f0"], 2),
        "hum_lines": len(info["lines"]),
        "hum_max_line_db": round(max(l["prom_db"] for l in info["lines"]), 1),
        "hum_gap_pct": round(100.0 * covered / len(x), 1),
    }
    return y, stats
