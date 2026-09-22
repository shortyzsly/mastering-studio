"""Dynamic bass control: evens out the 70-280 Hz region.

"Tight" bass is bass whose level doesn't swell and boom from word to word.
Proximity effect, plosive thumps and room resonance all make the low band
surge on certain syllables. A static EQ can only make the low end quieter
everywhere; this stage instead compresses just that band when it gets loud
*relative to this speaker's own low-band level* (a multiband-compressor /
dynamic-EQ move), so the steady body of the voice is left alone and the
surges are pulled back.

The band is isolated, gain-reduced, and subtracted from the full signal
(y = x - band * (1 - gain)) so nothing outside ~70-280 Hz changes. Reduction
is applied on speech frames only, never in the gaps.

Strength presets (threshold percentile / ratio / max reduction):
  light 80 / 2:1 / 3 dB    medium 70 / 3:1 / 5 dB    strong 60 / 4:1 / 8 dB
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import activity as act_mod
from . import blocks

STRENGTHS = {
    "light": (80.0, 2.0, 3.0),
    "medium": (70.0, 3.0, 5.0),
    "strong": (60.0, 4.0, 8.0),
}
# 70-280 Hz, not 70-200: for a voice with F0 ~100-140 Hz the 2nd harmonic (~200-280 Hz, where the
# 'mud' peak sits) swells with the fundamental. Watching only 70-200 left it uncontrolled. Measured swell
# (p95 minus median band level, speech frames): 7.8 dB with 70-200 vs 6.6 dB with 70-280 at Medium.
BAND_HZ = (70.0, 280.0)
BLOCK_MS = 2.0
ATTACK_MS = 6.0
RELEASE_MS = 90.0


def tighten(x: np.ndarray, sr: int, strength: str = "medium") -> tuple[np.ndarray, dict]:
    """x is (n,) or (n, ch); channels share one gain. Returns (audio, stats)."""
    percentile, ratio, max_db = STRENGTHS.get(strength, STRENGTHS["medium"])
    chans = x[:, None] if x.ndim == 1 else x
    mono = chans[:, 0] if chans.shape[1] == 1 else chans.mean(axis=1)
    act = act_mod.analyze_activity(mono, sr)

    sos = signal.butter(4, BAND_HZ, btype="bandpass", fs=sr, output="sos")
    block = max(1, int(sr * BLOCK_MS / 1000))
    nb = len(chans) // block
    if nb < 200:
        return x, {"bass_max_db": 0.0}
    lv = blocks.block_power_db(chans, block, sos, sr)

    per_hop = max(1, int(round(act.hop / block)))
    speech = np.repeat(act.speech_mask, per_hop)[:nb]
    speech = np.pad(speech, (0, nb - len(speech)))
    if speech.sum() < 200:
        return x, {"bass_max_db": 0.0}

    thr = float(np.percentile(lv[speech], percentile))
    gr = np.clip((lv - thr) * (1.0 - 1.0 / ratio), 0.0, max_db)
    gr[~speech] = 0.0

    a = np.exp(-BLOCK_MS / ATTACK_MS)
    r = np.exp(-BLOCK_MS / RELEASE_MS)
    sm = np.empty_like(gr)
    prev = 0.0
    for i, g in enumerate(gr):
        c = a if g > prev else r
        prev = c * prev + (1 - c) * g
        sm[i] = prev

    centers = (np.arange(nb) + 0.5) * block
    out = blocks.subtract_band_with_gain(x, sos, sr, centers, sm)
    stats = {
        "bass_max_db": round(float(sm.max()), 1),
        "bass_mean_db_on_speech": round(float(sm[speech].mean()), 1),
        "bass_active_pct": round(float((sm > 0.5).mean() * 100), 1),
    }
    return out, stats
