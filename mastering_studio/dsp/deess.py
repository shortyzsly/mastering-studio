"""Split-band de-esser: tames sibilance (the "s"/"sh" energy that reads as
harshness) without dulling the rest of the voice.

Two things adapt to the voice, so nothing is a fixed magic number:
  * WHERE: the band is centred on this speaker's own sibilant peak (found
    from the spectrum of their sibilant-heaviest frames, typically 5-9 kHz;
    lower for deeper voices), +/- ~0.4 octave.
  * WHEN: the threshold is a percentile of that band's level over speech
    frames, so only the loudest sibilants act.

Strength presets (percentile / ratio / max reduction):
  light 95 / 2:1 / 4 dB    medium 92 / 2.5:1 / 6 dB    strong 88 / 3:1 / 9 dB

Because reduction is applied to the isolated band and subtracted from the
full signal (y = x - band * (1 - gain)), everything outside the band is
bit-identical, unlike a wideband ducker which dulls the whole vowel.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import activity as act_mod
from . import blocks

BLOCK_MS = 1.0
STRENGTHS = {
    "light": (95.0, 2.0, 4.0),
    "medium": (92.0, 2.5, 6.0),
    "strong": (88.0, 3.0, 9.0),
}
SEARCH_LO_HZ, SEARCH_HI_HZ = 3500.0, 11000.0


def find_sibilance_band(x: np.ndarray, sr: int, act: act_mod.Activity) -> tuple[float, float]:
    """(lo_hz, hi_hz) centred on this voice's sibilance."""
    n_fft = 2048
    win = np.hanning(n_fft).astype(np.float32)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    hi_lim = min(SEARCH_HI_HZ, 0.45 * sr)
    sel = (f >= SEARCH_LO_HZ) & (f <= hi_lim)
    spectra, ratios = [], []
    for a, b in act_mod.frame_ranges(act.speech_mask, act.hop, act.frame, len(x)):
        for s in range(a, b - n_fft, n_fft // 2):
            p = np.abs(np.fft.rfft(x[s : s + n_fft] * win)) ** 2
            spectra.append(p)
            ratios.append(p[sel].sum() / (p.sum() + 1e-20))
        if len(spectra) > 30000:
            break
    if len(spectra) < 50:
        return 5000.0, min(10000.0, 0.45 * sr)
    top = np.argsort(ratios)[-max(10, len(ratios) // 25):]   # sibilant-heaviest ~4% of frames
    mean_p = np.mean([spectra[i] for i in top], axis=0)
    centroid = float((mean_p[sel] * f[sel]).sum() / mean_p[sel].sum())
    lo, hi = centroid / 1.35, centroid * 1.35
    return float(max(lo, 3500.0)), float(min(hi, 0.45 * sr))


def deess(
    x: np.ndarray,
    sr: int,
    strength: str = "medium",
    lo_hz: float | None = None,
    hi_hz: float | None = None,
    table: dict | None = None,
    attack_ms: float = 1.5,
    release_ms: float = 30.0,
    prefix: str = "deess",
) -> tuple[np.ndarray, dict]:
    """x is (n,) or (n, ch); channels share one gain. Returns (audio, stats).

    `table`, `attack_ms`, `release_ms` and `prefix` let other stages (e.g. the
    2-4.4 kHz presence control in dsp.soften) reuse this split-band compressor."""
    table = table or STRENGTHS
    percentile, ratio, max_reduction_db = table.get(strength, table["medium"])
    chans = x[:, None] if x.ndim == 1 else x
    mono = chans[:, 0] if chans.shape[1] == 1 else chans.mean(axis=1)  # NOT abs(): rectifying distorts the spectrum
    act = act_mod.analyze_activity(mono, sr)
    if lo_hz is None or hi_hz is None:
        lo_hz, hi_hz = find_sibilance_band(mono, sr, act)
    hi_hz = min(hi_hz, 0.45 * sr)
    sos = signal.butter(4, [lo_hz, hi_hz], btype="bandpass", fs=sr, output="sos")
    block = max(1, int(sr * BLOCK_MS / 1000))
    nb = len(chans) // block
    if nb < 100:
        return x, {f"{prefix}_max_db": 0.0}
    lv = blocks.block_power_db(chans, block, sos, sr)
    lv = np.convolve(lv, np.ones(3) / 3, mode="same")

    per_hop = max(1, act.hop // block)
    speech_blk = np.repeat(act.speech_mask, per_hop)[:nb]
    speech_blk = np.pad(speech_blk, (0, nb - len(speech_blk)))
    if speech_blk.sum() < 200:
        return x, {f"{prefix}_max_db": 0.0}

    thr = float(np.percentile(lv[speech_blk], percentile))
    gr = np.clip((lv - thr) * (1.0 - 1.0 / ratio), 0.0, max_reduction_db)
    gr[~speech_blk] = 0.0  # never touch the gaps

    a = np.exp(-BLOCK_MS / attack_ms)
    r = np.exp(-BLOCK_MS / release_ms)
    sm = np.empty_like(gr)
    prev = 0.0
    for i, g in enumerate(gr):
        c = a if g > prev else r
        prev = c * prev + (1 - c) * g
        sm[i] = prev

    centers = (np.arange(nb) + 0.5) * block
    out = blocks.subtract_band_with_gain(x, sos, sr, centers, sm)
    stats = {
        f"{prefix}_band_khz": (round(lo_hz / 1000, 1), round(hi_hz / 1000, 1)),
        f"{prefix}_threshold_db": round(thr, 1),
        f"{prefix}_max_db": round(float(sm.max()), 1),
        f"{prefix}_active_pct": round(float((sm > 0.5).mean() * 100), 1),
    }
    return out, stats
