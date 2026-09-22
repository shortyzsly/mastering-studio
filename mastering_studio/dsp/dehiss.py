"""HF de-hiss: a downward expander on the top octaves, for hiss under words.

Why this exists. The spectral denoiser (dsp.denoise) is a Wiener-type filter:
in a time-frequency bin where the voice is 10 dB above the noise it passes the
bin (gain ~0.9), because cutting it would cut the voice by the same amount. So
it removes essentially NO noise under speech (measured: -0.1 to -0.6 dB below
10 kHz), and no setting changes that -- depth, over-subtraction and estimator
memory all leave exposed-bin attenuation at about -2 dB. Noise under speech is
inaudible where the voice is loud (SNR 38-55 dB per band), and audible only in
soft syllables and word edges in the top octaves (SNR 11 dB at 6-10 kHz, 4 dB
at 10-16 kHz).

Placement matters: it runs BEFORE the denoiser, where this file's room-tone
level in each band is the true hiss level. (Measured after denoising, the room
tone is ~13 dB lower than the hiss that still sits under the words, so every
soft syllable looked far above threshold and nothing was expanded -- the first
version of this stage did exactly that.)

What this does about it. Speech at those frequencies is intermittent and much
louder than the hiss when it is real (sibilants and fricatives are 30+ dB above
it); when the top-octave level falls toward the hiss level, what is left is
mostly hiss. So each of two bands (5.5-9.5 kHz, 9.5 kHz-Nyquist) is expanded
downward below a threshold set a fixed number of dB above that band's own
room-tone level, measured from this file: the lower the band level, the more
it is pulled down, up to a cap. Loud consonants are above threshold and are not
touched. Attack is fast with 3 ms look-ahead so consonant onsets pass; release
is slow so the hiss is not pumped. The band is isolated, reduced and
subtracted from the full signal, so everything below 5.5 kHz is bit-identical.

Strengths (threshold above room tone / expansion ratio / max reduction):
  light +12 dB / 1.7:1 / 6 dB    medium +16 dB / 2.2:1 / 10 dB    strong +20 dB / 3:1 / 14 dB
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import activity as act_mod
from . import blocks

STRENGTHS = {
    "light": (12.0, 1.7, 6.0),
    "medium": (16.0, 2.2, 10.0),
    "strong": (20.0, 3.0, 14.0),
}
BANDS_HZ = ((5500.0, 9500.0), (9500.0, None))   # None => 0.45 * sr
BLOCK_MS = 1.0
LOOKAHEAD_MS = 3.0
OPEN_MS = 2.0       # gain rises (band gets louder): fast
CLOSE_MS = 90.0     # gain falls (band gets quieter): slow, no pumping


def _band_sos(lo: float, hi: float | None, sr: int) -> np.ndarray:
    hi = min(hi if hi is not None else 0.45 * sr, 0.45 * sr)
    return signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")


def dehiss(x: np.ndarray, sr: int, strength: str = "medium", act: act_mod.Activity | None = None) -> tuple[np.ndarray, dict]:
    """x is (n,) or (n, ch); channels share one gain per band. Returns (audio, stats)."""
    offset_db, ratio, max_db = STRENGTHS.get(strength, STRENGTHS["medium"])
    chans = x[:, None] if x.ndim == 1 else x
    mono = chans[:, 0] if chans.shape[1] == 1 else chans.mean(axis=1)
    act = act or act_mod.analyze_activity(mono, sr)

    block = max(1, int(sr * BLOCK_MS / 1000))
    nb = len(chans) // block
    if nb < 200 or 0.45 * sr <= BANDS_HZ[0][0] + 500:
        return x, {}
    per_hop = max(1, int(round(act.hop / block)))
    noise = np.pad(np.repeat(act.noise_mask, per_hop)[:nb], (0, max(0, nb - len(act.noise_mask) * per_hop)))[:nb]
    speech = np.pad(np.repeat(act.speech_mask, per_hop)[:nb], (0, max(0, nb - len(act.speech_mask) * per_hop)))[:nb]
    if noise.sum() < 100:
        return x, {}

    look = max(1, int(LOOKAHEAD_MS / BLOCK_MS))
    a = np.exp(-BLOCK_MS / OPEN_MS)
    r = np.exp(-BLOCK_MS / CLOSE_MS)
    centers = (np.arange(nb) + 0.5) * block
    out = x
    stats: dict = {}
    for lo, hi in BANDS_HZ:
        sos = _band_sos(lo, hi, sr)
        lv = blocks.block_power_db(out, block, sos, sr)
        lv = np.convolve(lv, np.ones(3) / 3, mode="same")                    # ~3 ms detector
        noise_lv = float(10 * np.log10(np.mean(10 ** (lv[noise] / 10.0))))   # this band's room tone
        thr = noise_lv + offset_db

        # Look-ahead: the gain must be open BEFORE a consonant arrives, so judge
        # each block by the loudest of itself and the next `look` blocks.
        lv_eff = lv.copy()
        for k in range(1, look + 1):
            lv_eff[:-k] = np.maximum(lv_eff[:-k], lv[k:])
        gr = np.clip((thr - lv_eff) * (ratio - 1.0), 0.0, max_db)

        sm = np.empty_like(gr)
        prev = 0.0
        for i, g in enumerate(gr):
            c = r if g > prev else a          # reduction growing = band closing = slow
            prev = c * prev + (1 - c) * g
            sm[i] = prev
        out = blocks.subtract_band_with_gain(out, sos, sr, centers, sm)

        tag = f"{lo/1000:g}k" if hi is None else f"{lo/1000:g}-{hi/1000:g}k"
        lin = 10 ** (-sm / 10.0)              # power domain: what happens to noise power
        stats[f"dehiss_{tag}_noise_db"] = round(noise_lv, 1)
        stats[f"dehiss_{tag}_under_speech_db"] = round(float(10 * np.log10(np.mean(lin[speech]))), 1) if speech.any() else 0.0
        loud = speech & (lv > thr + 10.0)
        stats[f"dehiss_{tag}_on_loud_consonants_db"] = round(float(10 * np.log10(np.mean(lin[loud]))), 2) if loud.any() else 0.0
    return out, stats
