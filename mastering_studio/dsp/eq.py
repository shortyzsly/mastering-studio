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
from scipy.ndimage import median_filter, uniform_filter1d

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
# max cut for the measured mud peak, shelf's upper transition Hz). All cuts --
# nothing is boosted. dsp.bass then handles SWELLS on top of this static
# shelf -- but dsp.bass is threshold-relative to the speaker's own median
# level, so a *constant* boom (small/untreated room, mic too close to a
# boundary) sits right through it unchanged; only this static shelf reaches
# that case. "max" exists for exactly that: a small vocal booth's boundary
# reinforcement is broader than one narrow resonance, often smearing 150-450
# Hz into a "muffled" character, so both the depth and the shelf's upper
# corner grow with strength.
LOWEND_PRESETS = {
    "light": (-0.5, -1.0, 2.5, 280.0),
    "medium": (-1.0, -2.0, 3.5, 320.0),
    "strong": (-2.0, -3.5, 5.0, 380.0),
    "max": (-3.0, -4.5, 6.5, 420.0),
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
    shelf, dip, cap, shelf_top_hz = LOWEND_PRESETS.get(strength, LOWEND_PRESETS["medium"])
    bells = [(MUD_DIP_CENTER_HZ, dip, MUD_DIP_SIGMA_OCT)]  # broad low-mid dip
    info: dict = {"low_shelf_db": shelf, "mud_dip_db": dip, "low_shelf_top_hz": shelf_top_hz}
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


def curve_db(
    freqs: np.ndarray,
    cuts: dict[float, float],
    amount: float = 1.0,
    warmth_db: float = 0.0,
    hf_tame: bool | str = False,
    low_shelf_db: float = 0.0,
    bells: list[tuple[float, float, float]] | None = None,
    low_shelf_top_hz: float = 320.0,
    shapes: list[tuple[str, float, float, float]] | None = None,
) -> np.ndarray:
    """Gain (dB) of the whole static EQ at `freqs`. Used both to build the FIR and to
    predict what the EQ will do to a measured spectrum (see tone_match)."""
    bells = bells or []
    logg = np.zeros(len(freqs))
    log_f = np.log2(np.maximum(np.asarray(freqs, dtype=np.float64), 1.0))
    for c, gain_db in cuts.items():
        # smooth bell, ~1 octave wide, in log-frequency
        logg += gain_db * amount * np.exp(-0.5 * ((log_f - np.log2(c)) / 0.35) ** 2)
    if low_shelf_db:
        # full effect below 150 Hz, none above low_shelf_top_hz, cosine transition in log-f
        t = np.clip((log_f - np.log2(150.0)) / (np.log2(low_shelf_top_hz) - np.log2(150.0)), 0.0, 1.0)
        logg += low_shelf_db * 0.5 * (1.0 + np.cos(np.pi * t))
    for c, gain_db, sigma in bells:
        logg += gain_db * np.exp(-0.5 * ((log_f - np.log2(c)) / sigma) ** 2)
    if warmth_db and amount > 0:
        logg += warmth_db * amount * np.exp(-0.5 * ((log_f - np.log2(WARMTH_CENTER_HZ)) / WARMTH_WIDTH_OCT) ** 2)
    if hf_tame and amount > 0:
        curve = HF_CURVES.get(hf_tame, HF_TAME) if isinstance(hf_tame, str) else HF_TAME
        fs, gs = zip(*curve)
        logg += amount * np.interp(log_f, np.log2(fs), gs)
    logg += shapes_db(log_f, shapes or [])
    return logg


def apply_curve(x: np.ndarray, sr: int, gain_db_fn) -> np.ndarray:
    """One linear-phase FIR whose magnitude is gain_db_fn(freqs) (dB)."""
    grid = np.linspace(0, sr / 2, 8193)   # firwin2's own grid size for NUM_TAPS: keeps narrow resonance notches exact
    g = gain_db_fn(grid)
    if not np.any(np.abs(g) > 0.01):
        return x
    fir = signal.firwin2(NUM_TAPS, grid / (sr / 2), 10 ** (g / 20.0))
    return blocks.fir_same_blocks(x, fir)


# --- parametric shapes: (kind, hz, gain_db, width_oct) ------------------------------------------------
# bell: Gaussian in log-frequency, width_oct = sigma. Shelves: smooth (logistic) transition about hz,
# width_oct = the octaves it takes to go from ~10% to ~90% of the gain.

def shapes_db(log_f: np.ndarray, shapes: list[tuple[str, float, float, float]]) -> np.ndarray:
    out = np.zeros_like(log_f)
    for kind, hz, gain, width in shapes:
        d = log_f - np.log2(hz)
        if kind == "bell":
            out += gain * np.exp(-0.5 * (d / width) ** 2)
        else:
            s = 1.0 / (1.0 + np.exp(-d * 4.4 / width))   # 0 below hz, 1 above
            out += gain * (s if kind == "high_shelf" else 1.0 - s)
    return out


# User EQ presets (GUI "Low / Mid / High"), applied on top of everything automatic.
EQ_PRESETS = {
    "low": {
        "off": [],
        "cut rumble": [("low_shelf", 90.0, -8.0, 0.6)],
        "cut boom": [("low_shelf", 140.0, -5.0, 0.8), ("bell", 110.0, -2.0, 0.35)],
        "warm boost": [("bell", 160.0, 2.0, 0.6)],
        "body boost": [("bell", 220.0, 1.5, 0.5)],
    },
    "mid": {
        "off": [],
        "cut mud": [("bell", 250.0, -3.0, 0.45)],
        "cut boxiness": [("bell", 500.0, -3.0, 0.4)],
        "cut nasal": [("bell", 1000.0, -2.5, 0.3)],
        "clarity boost": [("bell", 1600.0, 2.0, 0.6)],
    },
    "high": {
        "off": [],
        "cut harshness": [("bell", 3500.0, -3.0, 0.45)],
        "cut sibilance": [("bell", 7000.0, -3.0, 0.35)],
        "presence boost": [("bell", 4000.0, 2.5, 0.6)],
        "air boost": [("high_shelf", 10000.0, 3.0, 1.0)],
        "crisp (presence + air)": [("bell", 4500.0, 1.5, 0.6), ("high_shelf", 10000.0, 2.0, 1.0)],
    },
}


def preset_shapes(low: str = "off", mid: str = "off", high: str = "off") -> list[tuple[str, float, float, float]]:
    return EQ_PRESETS["low"].get(low, []) + EQ_PRESETS["mid"].get(mid, []) + EQ_PRESETS["high"].get(high, [])


# --- tone match: bring the voice's long-term balance toward a finished-narration target ----------------
# Target 1/3-octave speech levels relative to 1 kHz: a "clear, close-miked narration" curve (the
# balance speech enhancers like Adobe Enhance / eMastered land on), NOT the natural far-field LTASS
# (Byrne et al. 1994). LTASS puts 250-500 Hz ~7 dB above 1 kHz and 2-5 kHz ~6-9 dB below it; as a
# target that kept a booth's 400 Hz box hump and pulled presence DOWN, so masters read muddy and dull.
# This one: tight below 125 Hz, a low-mid plateau only ~2 dB over 1 kHz, presence held up (-2.5..-3.5 dB
# at 2-4 kHz; harshness and sibilance PEAKS are the dynamic stages' job, after this), a natural roll-off above 6 kHz. THE calibration knob for the automatic EQ: if finished
# files sound consistently too bright/dull/thin/muddy by ear, move these numbers.
TONE_TARGET_REL_1K = {
    80: -8.0, 100: -5.0, 125: -2.5, 160: 0.0, 200: 1.5, 250: 2.0, 315: 2.0, 400: 1.5, 500: 1.0,
    630: 1.0, 800: 0.5, 1000: 0.0, 1250: -1.0, 1600: -2.0, 2000: -2.5, 2500: -3.0, 3150: -3.0,
    4000: -3.5, 5000: -5.0, 6300: -6.5, 8000: -8.0, 10000: -10.0, 12500: -13.0,
}
TONE_CENTERS = list(TONE_TARGET_REL_1K)
TONE_DEADBAND_DB = 1.0   # differences inside this are a voice's character, not a problem
TONE_FRACTION = 0.85     # per pass; neighbouring bells overlap (~1.4x on a broad region), and the
TONE_PASSES = 3          # pipeline re-measures and corrects the residual (x0.3 per pass), so 3 passes land on target
# (lo_hz, hi_hz, max_cut_db, max_boost_db). Up to 520 Hz cut-only: boosting boom or low-mids re-adds the
# room's weight the notches just removed (by ear, +1.5 dB there read as "low end heavy"). Presence/air may be lifted.
TONE_LIMITS = [(0, 170, 10.0, 0.0), (170, 520, 8.0, 0.0), (520, 1600, 4.0, 1.5), (1600, 6000, 4.0, 4.0), (6000, 20000, 4.0, 2.0)]
# A boost lifts the band's noise with the voice: full boost only where the voice is >= 40 dB above the
# gaps in that band, none below 30 dB (a hissy file's presence/air is mostly noise; lifting it fails ACX).
BOOST_SNR_DB = (30.0, 40.0)


def tone_match(band_db: np.ndarray, planned_db: np.ndarray | float = 0.0, strength: float = 1.0,
               max_boost_scale: float = 1.0, min_hz: float = 0.0,
               applied: dict[int, float] | None = None, snr_db: np.ndarray | None = None) -> tuple[list[tuple[str, float, float, float]], dict]:
    """band_db: measured speech level per TONE_CENTERS band; planned_db: what the rest of the
    EQ already does there. Returns correction bells for what is still off-target (so this never
    stacks on the other EQ stages: it only adds the residual), plus a report."""
    pred = band_db + planned_db
    centers = np.array(TONE_CENTERS, dtype=float)
    target = np.array([TONE_TARGET_REL_1K[c] for c in TONE_CENTERS])
    ok = np.isfinite(pred)
    ref = (centers >= 800) & (centers <= 1250) & ok
    if ref.sum() < 2 or ok.sum() < 12:
        return [], {}
    diff = (pred - np.mean(pred[ref])) - (target - np.mean(target[ref]))   # + = too loud vs target
    shapes, report = [], {}
    snr_ok = np.ones(len(centers)) if snr_db is None else np.clip(
        (np.nan_to_num(snr_db, nan=0.0) - BOOST_SNR_DB[0]) / (BOOST_SNR_DB[1] - BOOST_SNR_DB[0]), 0.0, 1.0)
    for c, d, good, k in zip(centers, diff, ok, snr_ok):
        if not good or c < min_hz or abs(d) <= TONE_DEADBAND_DB:
            continue
        cut_cap, boost_cap = next((cu, bo * k) for lo, hi, cu, bo in TONE_LIMITS if lo <= c < hi)
        g = -np.sign(d) * (abs(d) - TONE_DEADBAND_DB) * TONE_FRACTION * strength
        done = (applied or {}).get(int(c), 0.0)   # earlier passes count against the caps
        g = float(np.clip(g, -cut_cap - done, boost_cap * max_boost_scale - done))
        if abs(g) >= 0.3:
            shapes.append(("bell", float(c), g, 0.19))   # ~1/3-octave bells, neighbours overlap smoothly
            report[int(c)] = round(g, 1)
    return shapes, {"tone_match_db": report} if report else {}


# --- room resonances (small-booth modes): narrow, fixed-frequency peaks that do NOT move with pitch --------
RES_RANGE_HZ = (80.0, 1000.0)
RES_MIN_SPEECH_S = 40.0   # below this the long-term spectrum still shows pitch harmonics; don't guess
RES_SEGMENTS = 8
RES_MIN_PROM_DB = 3.5   # median over segments; clean synthetic voices (male/female/monotone) stay below it
RES_CONSISTENCY = 0.75    # peak present in >= 75% of time segments (a harmonic drifts with pitch, a room mode doesn't)
RES_MAX = 4
RES_SMOOTH_PTS = 5      # 5/48 octave


def median_f0(x: np.ndarray, sr: int, activity: act_mod.Activity, max_frames: int = 1500) -> float | None:
    """Median pitch (Hz) of clearly voiced speech frames, by autocorrelation."""
    n = int(0.04 * sr)
    lo, hi = int(sr / 400), int(sr / 60)
    idx = np.flatnonzero(activity.speech_mask)
    if len(idx) == 0:
        return None
    f0s = []
    for j in idx[:: max(1, len(idx) // max_frames)]:
        seg = np.asarray(x[j * activity.hop : j * activity.hop + n], dtype=np.float64)
        if len(seg) < n:
            continue
        seg = seg - seg.mean()
        r = np.correlate(seg, seg, mode="full")[n - 1 :]
        if r[0] <= 0:
            continue
        r = r / r[0]
        k = lo + int(np.argmax(r[lo:hi]))
        if r[k] > 0.5:
            f0s.append(sr / k)
    return float(np.median(f0s)) if len(f0s) >= 20 else None


def find_resonances(x: np.ndarray, sr: int, activity: act_mod.Activity, strength: float = 1.0,
                    max_cut_db: float = 8.0) -> tuple[list[tuple[str, float, float, float]], dict]:
    n_fft = 1 << int(np.ceil(np.log2(sr / 6.0)))   # ~5 Hz bins (8192 @ 44.1/48k)
    hop_f = activity.hop
    sm = activity.speech_mask.astype(np.float64)
    span = max(1, n_fft // hop_f)
    frac = np.convolve(sm, np.ones(span) / span, mode="valid")   # speech fraction of the frame starting at each hop
    starts = [i * hop_f for i in range(0, len(frac), max(1, span // 2)) if frac[i] >= 0.6 and i * hop_f + n_fft <= len(x)]
    speech_s = len(starts) * n_fft / 2 / sr
    if speech_s < RES_MIN_SPEECH_S or len(starts) < RES_SEGMENTS * 4:
        return [], {"resonances": f"skipped - needs ~{RES_MIN_SPEECH_S:.0f} s of speech, file has {speech_s:.0f} s"}
    win = np.hanning(n_fft).astype(np.float32)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    logf_grid = np.arange(np.log2(RES_RANGE_HZ[0] / 1.5), np.log2(RES_RANGE_HZ[1] * 1.5), 1 / 48)   # 1/48 octave
    proms = []
    for seg in np.array_split(np.array(starts), RES_SEGMENTS):
        acc = np.zeros(len(f))
        for s in seg:
            acc += np.abs(np.fft.rfft(x[s : s + n_fft] * win)) ** 2
        db = np.interp(logf_grid, np.log2(np.maximum(f, 1.0)), 10 * np.log10(acc / len(seg) + 1e-20))
        db = uniform_filter1d(db, RES_SMOOTH_PTS, mode="nearest")   # a harmonic voice's fine spectrum is spiky
        proms.append(db - median_filter(db, size=33, mode="nearest"))   # vs the local +/-1/3 octave level
    proms = np.array(proms)
    mean_p = np.median(proms, axis=0)   # robust to one odd segment
    consistent = (proms > RES_MIN_PROM_DB * 0.6).mean(axis=0)
    in_range = (logf_grid >= np.log2(RES_RANGE_HZ[0])) & (logf_grid <= np.log2(RES_RANGE_HZ[1]))
    peaks = [i for i in range(1, len(mean_p) - 1)
             if in_range[i] and mean_p[i] >= mean_p[i - 1] and mean_p[i] > mean_p[i + 1]
             and mean_p[i] >= RES_MIN_PROM_DB and consistent[i] >= RES_CONSISTENCY]
    # A monotone narrator's pitch harmonics stay put too; anything on k*F0 is the voice, not the room.
    f0 = median_f0(x, sr, activity)
    if f0:
        peaks = [i for i in peaks if min(abs(2 ** logf_grid[i] / (k * f0) - 1) for k in range(1, 12)) > 0.035]
    peaks = sorted(peaks, key=lambda i: -mean_p[i])[:RES_MAX]
    shapes, found = [], []
    for i in sorted(peaks):
        half = mean_p[i] / 2
        lo = i
        while lo > 0 and mean_p[lo] > half:
            lo -= 1
        hi = i
        while hi < len(mean_p) - 1 and mean_p[hi] > half:
            hi += 1
        sigma = float(np.clip((hi - lo) / 48 / 2.355, 1 / 40, 1 / 6))
        hz = float(2 ** logf_grid[i])
        cut = -float(min(max_cut_db, mean_p[i] * strength))   # the whole measured peak: a room mode is never 'character'
        shapes.append(("bell", hz, cut, sigma))
        found.append({"hz": int(round(hz)), "prominence_db": round(float(mean_p[i]), 1), "cut_db": round(cut, 1),
                      "q": round(1 / (2 * np.sinh(np.log(2) / 2 * 2.355 * sigma)), 1)})
    return shapes, {"resonances": found, "voice_f0_hz": round(f0) if f0 else None}
