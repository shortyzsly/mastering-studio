"""De-breath: attenuate inhale/exhale noises between phrases, never remove them.

Engineers reduce breaths by roughly 6-14 dB rather than deleting them
(iZotope RX "Breath Control", Waves DeBreath, Auphonic "Remove Breaths"):
a narration with silent gaps where a breath should be sounds robotic, and
ACX rejects digital silence anyway. So this stage finds breath events and
ducks them by a bounded amount, and never below the room tone.

A frame belongs to a breath only if ALL of these hold:
  * UNVOICED: normalized autocorrelation peak in the pitch-lag range is low.
    Breath is turbulent noise; vowels/voiced consonants are periodic, so
    words (including quiet ones) are excluded by this test.
  * LEVEL: between ~12 and ~34 dB below the active-speech level (and 10 dB
    above room tone). Louder = fricatives (s, sh, f); fainter = decay tails.
  * DURATION: the merged run lasts 240 ms - 1.5 s. Fricatives, plosive bursts
    and word-final tails are shorter.
  * POSITION: the event STARTS FROM ROOM TONE (a pause right before it), so
    it is an onset out of silence. Word/phrase-final tails do the opposite:
    they start at speech level and decay into the pause, and are excluded.

Tuned by looking at spectrograms of every class of flagged event on real
narration (see README): the inhales that pass these tests are broadband,
harmonic-free noise of 0.25-0.45 s leading straight into a phrase; the
events that were rejected were speech decay tails and word endings.

Anything that fails a test is left bit-identical. Gain changes ramp in and
out *inside* the event, so the word that follows is never touched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import activity as act_mod
from . import blocks

STRENGTH_DB = {"light": 6.0, "medium": 10.0, "strong": 14.0}

VOICED_MAX = 0.35        # autocorr peak below this => unvoiced
MIN_ABOVE_FLOOR_DB = 10.0
MAX_BELOW_SPEECH_DB = 12.0   # louder than speech-12 dB => not a breath
MIN_BELOW_SPEECH_DB = 34.0   # fainter than speech-34 dB => decay tail, leave alone
MIN_DURATION_S = 0.24
MAX_DURATION_S = 1.5
CLOSE_GAP_FRAMES = 3     # bridge <=30 ms dropouts inside one breath
GROW_FRAMES = 10         # onset/decay ramps may extend a run by up to 100 ms each side
GROW_ABOVE_FLOOR_DB = 8.0
GROW_VOICED_MAX = 0.45   # slightly looser on the ramps, where the signal is faint
MIN_SEED_FRAMES = 8      # a real event has >= 80 ms of clearly-qualifying frames
PAUSE_ABOVE_FLOOR_DB = 9.0
MAX_HF_FRACTION = 0.40   # >40% of energy above 5 kHz => sibilant onset (s/sh), not a breath
PAUSE_LOOK_FRAMES = 10   # 100 ms either side
KEEP_ABOVE_FLOOR_DB = 3.0  # never duck a breath below floor + this


@dataclass
class BreathEvent:
    start: int       # sample
    end: int         # sample
    level_db: float  # median frame level
    reduction_db: float


def _harmonicity(x: np.ndarray, sr: int, act: act_mod.Activity, frames: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation peak (pitch range 70-400 Hz) per frame index
    in `frames`; 40 ms Hann window. Computed only for the frames asked for."""
    win_len = int(0.04 * sr)
    n_fft = 1 << int(np.ceil(np.log2(win_len * 2)))
    lo, hi = int(sr / 400), int(sr / 70)
    win = np.hanning(win_len).astype(np.float32)
    out = np.ones(len(frames))
    for chunk in range(0, len(frames), 2048):
        idx = frames[chunk : chunk + 2048]
        starts = idx * act.hop - (win_len - act.frame) // 2
        ok = (starts >= 0) & (starts + win_len <= len(x))
        if not ok.any():
            continue
        seg = np.stack([x[s : s + win_len] for s in starts[ok]]) * win
        power = np.abs(np.fft.rfft(seg, n_fft, axis=1)) ** 2
        ac = np.fft.irfft(power, axis=1)
        h = ac[:, lo:hi].max(axis=1) / (ac[:, 0] + 1e-20)
        out[chunk : chunk + 2048][ok] = h
    return out


def _hf_fraction(x: np.ndarray, sr: int, s: int, e: int) -> float:
    seg = x[s : min(e, len(x))]
    if len(seg) < 512:
        return 0.0
    n_fft = 2048
    win = np.hanning(min(n_fft, len(seg))).astype(np.float32)
    acc = 0.0
    for i in range(0, max(1, len(seg) - len(win) + 1), len(win) // 2):
        acc = acc + np.abs(np.fft.rfft(seg[i : i + len(win)] * win, n_fft)) ** 2
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    return float(acc[f >= 5000].sum() / (acc.sum() + 1e-20))


def _grow(seeds: np.ndarray, allowed: np.ndarray, max_frames: int) -> np.ndarray:
    """Hysteresis: extend seed runs outward (<= max_frames each side) through
    frames that are still plausible breath (`allowed`)."""
    cur = seeds.copy()
    for _ in range(max_frames):
        nxt = cur.copy()
        nxt[1:] |= cur[:-1]
        nxt[:-1] |= cur[1:]
        nxt &= allowed | cur
        if (nxt == cur).all():
            break
        cur = nxt
    return cur


def detect(x: np.ndarray, sr: int, act: act_mod.Activity | None = None) -> list[tuple[int, int]]:
    """Return breath events as frame-index runs [(first, last_exclusive)].

    Seeds are frames that clearly qualify (level window + unvoiced). Runs are
    then grown through the soft onset/decay ramps (still unvoiced, still above
    room tone), because a breath swells and fades: judging only its loud core
    would cut it off mid-ramp and make the "starts from a pause" test look at
    the ramp instead of the silence before it.
    """
    act = act or act_mod.analyze_activity(x, sr)
    lv = act.level_db
    n = len(lv)
    lo_db = max(act.floor_db + MIN_ABOVE_FLOOR_DB, act.speech_db - MIN_BELOW_SPEECH_DB)
    hi_db = act.speech_db - MAX_BELOW_SPEECH_DB
    in_window = (lv >= lo_db) & (lv <= hi_db)
    ramp_ok = (lv >= act.floor_db + GROW_ABOVE_FLOOR_DB) & (lv <= hi_db + 3.0)

    hval = np.ones(n)
    need = np.flatnonzero(in_window | ramp_ok)
    if len(need):
        hval[need] = _harmonicity(x, sr, act, need)
    seeds = in_window & (hval < VOICED_MAX)
    grown = _grow(seeds, ramp_ok & (hval < GROW_VOICED_MAX), GROW_FRAMES)
    if CLOSE_GAP_FRAMES:
        closed = np.convolve(grown.astype(float), np.ones(CLOSE_GAP_FRAMES + 1), "same") > 0
        closed &= ramp_ok | grown
    else:
        closed = grown

    pad = np.concatenate(([False], closed, [False]))
    edges = np.flatnonzero(np.diff(pad.astype(np.int8)))
    min_f = int(MIN_DURATION_S * 1000 / 10)
    max_f = int(MAX_DURATION_S * 1000 / 10)

    quiet = lv <= act.floor_db + PAUSE_ABOVE_FLOOR_DB
    events = []
    for a, b in zip(edges[::2], edges[1::2]):
        if not (min_f <= b - a <= max_f):
            continue
        if seeds[a:b].sum() < MIN_SEED_FRAMES:
            continue
        before = quiet[max(0, a - PAUSE_LOOK_FRAMES) : a]
        if not (len(before) and before.mean() >= 0.5):   # must start out of a pause
            continue
        if _hf_fraction(x, sr, a * act.hop, b * act.hop) > MAX_HF_FRACTION:
            continue
        events.append((int(a), int(b)))
    return events


def debreath(
    x: np.ndarray,
    sr: int,
    reduction_db: float = 10.0,
    act: act_mod.Activity | None = None,
) -> tuple[np.ndarray, list[BreathEvent]]:
    """x is (n,) or (n, ch); one linked gain for all channels."""
    mono = x if x.ndim == 1 else x.mean(axis=1)  # detection signal; NOT abs() (rectifying fakes periodicity)
    act = act or act_mod.analyze_activity(mono, sr)
    runs = detect(mono, sr, act)
    if not runs or reduction_db <= 0:
        return x, []

    n_frames = len(act.level_db)
    gain_db = np.zeros(n_frames)
    events: list[BreathEvent] = []
    for a, b in runs:
        level = float(np.median(act.level_db[a:b]))
        red = float(np.clip(min(reduction_db, level - (act.floor_db + KEEP_ABOVE_FLOOR_DB)), 0.0, reduction_db))
        if red < 1.0:
            continue
        gain_db[a:b] = -red
        events.append(BreathEvent(a * act.hop, b * act.hop + act.frame, level, red))
    if not events:
        return x, []

    # Soft edges: a 3-frame ramp (~30 ms) at each end, INSIDE the event, so the
    # duck fades in/out without ever reaching the speech that follows.
    smooth = gain_db.copy()
    for a, b in runs:
        if gain_db[a] == 0.0:
            continue
        idx = np.arange(a, b)
        ramp = np.minimum(1.0, np.minimum(idx - a + 1, b - idx) / 3.0)
        smooth[a:b] = gain_db[a:b] * ramp

    centers = act.frame / 2 + np.arange(n_frames) * act.hop
    return blocks.apply_gain_db(x, centers, smooth), events
