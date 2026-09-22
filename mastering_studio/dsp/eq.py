"""Subtractive corrective EQ, measured on speech only.

What the first version got wrong: it compared the whole-file band balance
(room tone included) against a hand-made "flat-ish" target. Real speech is
NOT flat -- it slopes down ~6-9 dB/octave above ~500 Hz -- so it "corrected"
normal speech by cutting the low-mids 4 dB, then ran the filter through
sosfiltfilt, which applies it forward *and* backward and so doubled every
correction (8 dB). That thinned the voice by ~4 dB before compression.

This version follows the usual engineer's rule -- cut problems, don't
re-shape a voice:
  * measure the spectrum of speech-active frames only, in 1/3-octave bands;
  * fit the voice's own smooth trend (robust, in dB vs log-frequency) and
    treat only *bumps above the trend* (boxiness, honk, harshness) as
    problems;
  * cut those bumps, max 3 dB each, never boost;
  * apply as a single-pass linear-phase FIR, so a 3 dB cut is exactly 3 dB.

On top of the measured cuts there is optional fixed "voice polish": a broad
warmth bell at 150 Hz (gives back the body the tighter high-pass removes) and
a top-end ease-off (HF_TAME) that reduces harshness/hiss above 6 kHz.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import activity as act_mod
from . import blocks

THIRD_OCT = 2 ** (1 / 3)
CENTERS = [125 * THIRD_OCT ** i for i in range(0, 20) if 125 * THIRD_OCT ** i <= 9000]
BUMP_THRESHOLD_DB = 3.0   # residual above trend before we act
MAX_CUT_DB = 2.5          # never cut more than this
NUM_TAPS = 6143   # finer low-frequency resolution for narrow bass/mud bells


def measure_speech_bands(
    x: np.ndarray, sr: int, activity: act_mod.Activity, centers: list[float] | None = None, octave_frac: float = 3.0
) -> np.ndarray:
    """Mean power (dB) of speech-active frames, per band of 1/octave_frac octave."""
    centers = CENTERS if centers is None else centers
    n_fft = 8192 if octave_frac > 3 else 4096
    win = np.hanning(n_fft).astype(np.float32)
    acc = np.zeros(n_fft // 2 + 1)
    count = 0
    for a, b in act_mod.frame_ranges(activity.speech_mask, activity.hop, activity.frame, len(x)):
        for s in range(a, b - n_fft, n_fft):
            spec = np.fft.rfft(x[s : s + n_fft] * win)
            acc += np.abs(spec) ** 2
            count += 1
            if count >= 6000:
                break
        if count >= 6000:
            break
    if count == 0:
        return np.full(len(centers), np.nan)
    psd = acc / count
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    out = []
    half = 2 ** (0.5 / octave_frac)
    for c in centers:
        m = (freqs >= c / half) & (freqs < c * half)
        out.append(10 * np.log10(psd[m].sum() + 1e-20) if m.any() else np.nan)
    return np.array(out)


# Low-end presets: (low-shelf dB below ~150 Hz, broad low-mid dip dB @320 Hz,
# max cut for the measured mud peak). All cuts -- nothing is boosted. The shelf
# is deliberately gentle: the dynamic bass stage (dsp.bass) does the rest of
# the tightening, and over-cutting the shelf makes a voice thin, not tight.
LOWEND_PRESETS = {
    "light": (-0.5, -1.0, 2.5),
    "medium": (-1.0, -2.0, 3.5),
    "strong": (-2.0, -3.5, 5.0),
}
MUD_PEAK_FRACTION = 0.85       # remove ~85% of a peak's excess over local level; keep some body
MUD_MAX_PEAKS = 2
MUD_PEAK_SEPARATION_OCT = 0.4
MUD_DIP_CENTER_HZ = 320.0
MUD_DIP_SIGMA_OCT = 0.6
LOWEND_OWNS_BELOW_HZ = 600.0   # generic auto-cuts below this are dropped when the low-end stage is on
FINE_CENTERS = [100 * 2 ** (k / 6) for k in range(0, 15)]   # 100 Hz .. ~560 Hz, 1/6 octave
MUD_SEARCH_HZ = (180.0, 520.0)
MUD_MIN_PEAK_DB = 1.5
MUD_BELL_SIGMA_OCT = 0.22


def lowend_plan(band_db: np.ndarray, strength: str) -> tuple[float, list[tuple[float, float, float]], dict]:
    """(low_shelf_db, bells, info). `band_db` is measure_speech_bands over
    FINE_CENTERS. The mud peak is found the way an engineer sweeps for it: the
    band that stands furthest above the *local* level (median of its
    neighbours within +/-0.75 octave), searched between 180 and 450 Hz."""
    shelf, dip, cap = LOWEND_PRESETS.get(strength, LOWEND_PRESETS["medium"])
    bells = [(MUD_DIP_CENTER_HZ, dip, MUD_DIP_SIGMA_OCT)]  # broad low-mid dip
    info: dict = {"low_shelf_db": shelf, "mud_dip_db": dip}
    ok = np.isfinite(band_db)
    if ok.sum() >= 8:
        logc = np.log2(np.array(FINE_CENTERS))
        resid = np.full(len(band_db), -np.inf)
        for i in range(len(band_db)):
            near = ok & (np.abs(logc - logc[i]) <= 0.75) & (np.arange(len(band_db)) != i)
            if near.sum() >= 3 and ok[i]:
                resid[i] = band_db[i] - np.median(band_db[near])
        search = np.array([MUD_SEARCH_HZ[0] <= c <= MUD_SEARCH_HZ[1] for c in FINE_CENTERS])
        cand = np.where(search, resid, -np.inf)
        peaks = []
        for _ in range(MUD_MAX_PEAKS):
            j = int(np.argmax(cand))
            if not (np.isfinite(cand[j]) and cand[j] >= MUD_MIN_PEAK_DB):
                break
            cut = -float(min(cap, cand[j] * MUD_PEAK_FRACTION))
            bells.append((FINE_CENTERS[j], cut, MUD_BELL_SIGMA_OCT))
            peaks.append({"hz": int(round(FINE_CENTERS[j])), "over_local_db": round(float(cand[j]), 1), "cut_db": round(cut, 1)})
            cand[np.abs(logc - logc[j]) < MUD_PEAK_SEPARATION_OCT] = -np.inf   # next peak must be elsewhere
        if peaks:
            info["mud_peaks"] = peaks
    return shelf, bells, info


def compute_cuts(band_db: np.ndarray) -> dict[float, float]:
    """Cut (negative dB) per band centre for bumps above the voice's trend."""
    ok = np.isfinite(band_db)
    if ok.sum() < 8:
        return {}
    logf = np.log2(np.array(CENTERS))
    # Trend over 200 Hz - 8 kHz, robust to bumps: fit, drop outliers, refit.
    use = ok & (np.array(CENTERS) >= 200) & (np.array(CENTERS) <= 8000)
    coef = np.polyfit(logf[use], band_db[use], 2)
    resid = band_db - np.polyval(coef, logf)
    keep = use & (resid < np.percentile(resid[use], 70))
    coef = np.polyfit(logf[keep], band_db[keep], 2)
    resid = band_db - np.polyval(coef, logf)

    cuts = {}
    for c, r, u in zip(CENTERS, resid, use):
        if u and r > BUMP_THRESHOLD_DB:
            cuts[c] = -float(min(MAX_CUT_DB, (r - BUMP_THRESHOLD_DB + 1.0) * 0.75))
    return cuts


# Fixed top-end shaping (Hz, dB): leaves <=4 kHz alone, eases the 6-10 kHz
# "edge", and rolls off the 12 kHz+ region where a voice has almost nothing
# but hiss/air noise. Gentle enough not to sound dull (~-3 dB at 8-10 kHz).
HF_TAME = [(4000, 0.0), (6300, -1.0), (8000, -2.5), (10000, -3.5), (14000, -6.0), (20000, -9.0)]
# Harshness presets replace HF_TAME (they include it). They start lower (the bass/mud
# cuts made everything above them relatively brighter) and go deeper up top, which
# also lowers hiss under the words in exactly the octaves where it is exposed.
HF_CURVES = {
    "base": HF_TAME,
    "light": [(3000, 0.0), (4000, -0.5), (5000, -1.0), (6300, -1.8), (8000, -3.0), (10000, -4.0), (14000, -6.5), (20000, -9.5)],
    "medium": [(2000, 0.0), (3000, -0.8), (4000, -1.5), (5000, -2.3), (6300, -3.2), (8000, -4.5), (10000, -5.5), (14000, -8.0), (20000, -11.5)],
    "strong": [(1500, 0.0), (2500, -1.0), (3500, -2.2), (5000, -3.5), (6300, -4.5), (8000, -6.0), (10000, -7.5), (14000, -10.0), (20000, -14.0)],
}
WARMTH_CENTER_HZ = 150.0
WARMTH_WIDTH_OCT = 0.7


def apply_cuts(
    x: np.ndarray,
    sr: int,
    cuts: dict[float, float],
    amount: float = 1.0,
    warmth_db: float = 0.0,
    hf_tame: bool | str = False,
    low_shelf_db: float = 0.0,
    bells: list[tuple[float, float, float]] | None = None,
) -> np.ndarray:
    bells = bells or []
    scaled = amount > 0 and bool(cuts or warmth_db or hf_tame)
    if not (scaled or low_shelf_db or bells):
        return x
    grid = np.linspace(0, sr / 2, 2048)
    logg = np.zeros_like(grid)
    log_f = np.log2(np.maximum(grid, 1.0))
    for c, gain_db in cuts.items():
        # smooth bell, ~1 octave wide, in log-frequency
        logg += gain_db * amount * np.exp(-0.5 * ((log_f - np.log2(c)) / 0.35) ** 2)
    if low_shelf_db:
        # full effect below 150 Hz, none above 320 Hz, cosine transition in log-f
        t = np.clip((log_f - np.log2(150.0)) / (np.log2(320.0) - np.log2(150.0)), 0.0, 1.0)
        logg += low_shelf_db * 0.5 * (1.0 + np.cos(np.pi * t))
    for c, gain_db, sigma in bells:
        logg += gain_db * np.exp(-0.5 * ((log_f - np.log2(c)) / sigma) ** 2)
    if warmth_db and amount > 0:
        logg += warmth_db * amount * np.exp(-0.5 * ((log_f - np.log2(WARMTH_CENTER_HZ)) / WARMTH_WIDTH_OCT) ** 2)
    if hf_tame and amount > 0:
        curve = HF_CURVES.get(hf_tame, HF_TAME) if isinstance(hf_tame, str) else HF_TAME
        fs, gs = zip(*curve)
        logg += amount * np.interp(log_f, np.log2(fs), gs)
    mag = 10 ** (logg / 20.0)
    fir = signal.firwin2(NUM_TAPS, grid / (sr / 2), mag)
    return blocks.fir_same_blocks(x, fir)
