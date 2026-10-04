"""Detect and duck brief, highly tonal transients that sit in inter-word
pauses -- a single near-pure "blip"/chime, as distinct from breath noise
(broadband), room tone (broadband, sustained) or normal speech (broadband
formant structure, not a near-pure tone).

Unlike `dsp.debreath`, this is NOT restricted to quiet levels: a real,
brief tonal event picked up by the mic during a pause (a notification
chime, a physical tap/ring, mic-adjacent electronics) can be as loud as, or
louder than, normal speech, and can even straddle the speech/gap boundary
in the activity detector because it fires "loud" for a moment (see the
project's Chapter 1 investigation: peak level up to -13 dBFS, well above
this narrator's own speech level, but with spectral flatness ~0.02, i.e.
close to a pure tone -- speech at that level is never that tonal). The one
thing that DOES separate it from a word is exactly that: real speech is
broadband/formant-rich even when loud, this is not.

These are real acoustic events, not something the DSP chain introduces
(confirmed on Chapter 1 of "The Knight Who Would Not Die": the exact same
tone, at the exact same timestamp, with matching level and spectral
content, is already present in the unprocessed source. Denoising and
loudness normalization only reveal them, by removing the broadband floor
that used to mask them).

STATUS: not wired into the pipeline. Threshold tuning on that same file
swung from ~800 false positives (catching ordinary quiet speech texture)
down to 2 events that didn't include either of the two confirmed
instances -- i.e. detection reliability isn't solid enough to run
unsupervised yet. Kept here as a documented starting point, not a shipped
fix. Needs either a better feature (e.g. harmonic-comb regularity, since a
real chime is often more evenly harmonic than a formant) or a human
listening pass on a labelled set of true positives/negatives to tune
against, neither of which is available from inside this environment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import activity as act_mod
from . import blocks

FLATNESS_MAX = 0.05          # spectral flatness below this = tonal (speech/breath/room tone essentially never get this low)
MIN_ABOVE_FLOOR_DB = 8.0     # must at least clear the room floor to be worth judging
MIN_DURATION_S = 0.05
MAX_DURATION_S = 1.5
ANALYSIS_WIN_S = 0.08        # window the flatness check is measured over, centred on each candidate peak
BOUNDARY_DROP_DB = 15.0      # event runs while level stays within this of its own peak
GAP_FRACTION_MIN = 0.6       # the analysis window must be mostly outside the speech mask
STRENGTH_DB = {"light": 12.0, "medium": 18.0, "strong": 24.0}


@dataclass
class BlipEvent:
    start: int
    end: int
    peak_db: float
    peak_freq_hz: float
    flatness: float


def _spectral_flatness_and_peak(seg: np.ndarray, sr: int, lo: float = 150.0, hi: float = 8000.0) -> tuple[float, float]:
    n = len(seg)
    if n < 64:
        return 1.0, 0.0
    win = np.hanning(n).astype(np.float32)
    spec = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    m = (freqs >= lo) & (freqs <= hi)
    if not m.any():
        return 1.0, 0.0
    p = spec[m].astype(np.float64) ** 2 + 1e-20
    flat = float(np.exp(np.mean(np.log(p))) / np.mean(p))
    peak_f = float(freqs[m][np.argmax(p)])
    return flat, peak_f


def detect(x: np.ndarray, sr: int, act: act_mod.Activity | None = None) -> list[BlipEvent]:
    act = act or act_mod.analyze_activity(x, sr)
    lv = act.level_db
    n = len(lv)
    win_n = int(ANALYSIS_WIN_S * sr)

    per_win = max(1, win_n // act.hop)

    events: list[BlipEvent] = []
    i = 2
    while i < n - 2:
        is_peak = (
            lv[i] > act.floor_db + MIN_ABOVE_FLOOR_DB
            and lv[i] > lv[i - 1] and lv[i] > lv[i + 1]
            and lv[i] > lv[i - 2] and lv[i] > lv[i + 2]
        )
        if is_peak:
            rise = lv[i] - float(np.min(lv[max(0, i - 4):i])) if i >= 1 else 0.0
            fall = lv[i] - float(np.min(lv[i + 1:i + 5]))
            gap_frac = 1.0 - float(act.speech_mask[max(0, i - per_win): i + per_win].mean())
            if rise > 6.0 and fall > 6.0 and gap_frac >= GAP_FRACTION_MIN:
                c = i * act.hop
                seg = x[max(0, c - win_n // 2): c + win_n // 2]
                flat, peak_f = _spectral_flatness_and_peak(seg, sr)
                if flat < FLATNESS_MAX:
                    peak_db = lv[i]
                    a = i
                    while a > 0 and lv[a - 1] > peak_db - BOUNDARY_DROP_DB:
                        a -= 1
                    b = i
                    while b < n - 1 and lv[b + 1] > peak_db - BOUNDARY_DROP_DB:
                        b += 1
                    dur = (b - a) * act.hop / sr
                    if MIN_DURATION_S <= dur <= MAX_DURATION_S:
                        events.append(BlipEvent(a, b, float(peak_db), peak_f, flat))
                        i = b + 1
                        continue
        i += 1

    # merge events that ended up adjacent/overlapping after boundary expansion
    merged: list[BlipEvent] = []
    for ev in sorted(events, key=lambda e: e.start):
        if merged and ev.start <= merged[-1].end + 5:
            prev = merged[-1]
            merged[-1] = BlipEvent(prev.start, max(prev.end, ev.end), max(prev.peak_db, ev.peak_db), prev.peak_freq_hz, min(prev.flatness, ev.flatness))
        else:
            merged.append(ev)
    return merged


def duck(
    x: np.ndarray, sr: int, reduction_db: float = 18.0, act: act_mod.Activity | None = None
) -> tuple[np.ndarray, list[BlipEvent]]:
    """x is (n,) or (n, ch); one linked gain for all channels."""
    mono = x if x.ndim == 1 else x.mean(axis=1)
    act = act or act_mod.analyze_activity(mono, sr)
    events = detect(mono, sr, act)
    if not events or reduction_db <= 0:
        return x, []

    n_frames = len(act.level_db)
    gain_db = np.zeros(n_frames)
    for ev in events:
        gain_db[ev.start:ev.end] = -reduction_db

    smooth = gain_db.copy()
    for ev in events:
        idx = np.arange(ev.start, ev.end)
        ramp = np.minimum(1.0, np.minimum(idx - ev.start + 1, ev.end - idx) / 3.0)
        smooth[ev.start:ev.end] = gain_db[ev.start:ev.end] * ramp

    centers = act.frame / 2 + np.arange(n_frames) * act.hop
    return blocks.apply_gain_db(x, centers, smooth), events
