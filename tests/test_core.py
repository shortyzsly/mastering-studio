"""Self-checks for the automatic decisions. Run: .venv/bin/python tests/test_core.py
Synthetic signals with known answers (a room of known T60, a known booth mode, known clicks),
so each check fails if its detector or fix breaks."""
import os
import sys
import tempfile

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mastering_studio import acx, io_utils  # noqa: E402
from mastering_studio.dsp import activity, declick, dereverb, eq, roomtone  # noqa: E402
from mastering_studio.pipeline import PipelineSettings, process_array  # noqa: E402
from mastering_studio.analysis import analyze  # noqa: E402

SR = 44100


def speechlike(seconds, seed=0, f0_range=(90, 150)):
    """Harmonic 'syllables' with gliding pitch, short gaps and some longer pauses."""
    rng = np.random.default_rng(seed)
    out, n_total = [], 0
    while n_total < seconds * SR:
        n = int(rng.uniform(0.15, 0.4) * SR)
        f0 = rng.uniform(*f0_range) * np.linspace(1, rng.uniform(0.85, 1.15), n)
        ph = 2 * np.pi * np.cumsum(f0) / SR
        syl = sum(np.sin(k * ph) / k for k in range(1, 40) if k * f0.max() < 8000) + 0.05 * rng.standard_normal(n)
        syl *= np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.02 * SR))
        gap = np.zeros(int((rng.uniform(0.05, 0.25) if rng.random() < 0.8 else rng.uniform(0.4, 0.9)) * SR))
        out += [syl, gap]
        n_total += n + len(gap)
    x = signal.lfilter(*signal.butter(1, 3000, fs=SR), np.concatenate(out))
    return (0.1 * x / np.sqrt(np.mean(x ** 2))).astype(np.float32)


def noise(n, db, seed=3):
    return (10 ** (db / 20) * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


def reverb(x, t60, drr_db=6.0):
    n = int(t60 * SR)
    tail = np.random.default_rng(1).standard_normal(n) * np.exp(-3 * np.log(10) * np.arange(n) / SR / t60)
    tail[: int(0.003 * SR)] = 0
    tail *= 10 ** (-drr_db / 20) / np.sqrt(np.sum(tail ** 2))
    tail[0] = 1.0
    return signal.fftconvolve(x, tail)[: len(x)].astype(np.float32)


def peaking(x, f, g, q):
    a_, w = 10 ** (g / 40), 2 * np.pi * f / SR
    al = np.sin(w) / (2 * q)
    return signal.lfilter([1 + al * a_, -2 * np.cos(w), 1 - al * a_], [1 + al / a_, -2 * np.cos(w), 1 - al / a_], x).astype(np.float32)


def test_tone_match():
    tgt = np.array([eq.TONE_TARGET_REL_1K[c] for c in eq.TONE_CENTERS])
    shapes, _ = eq.tone_match(tgt + 1.0)                     # on target (any overall offset): nothing to do
    assert shapes == [], shapes
    boomy = tgt.copy()
    boomy[eq.TONE_CENTERS.index(100)] += 10.0
    shapes, info = eq.tone_match(boomy)
    cut = info["tone_match_db"][100]
    assert abs(cut - (-(10 - eq.TONE_DEADBAND_DB) * eq.TONE_FRACTION)) < 0.2, cut
    assert all(c == 100 for _, c, _, _ in shapes)
    thin = tgt.copy()
    thin[eq.TONE_CENTERS.index(100)] -= 10.0                 # lows are cut-only: never boost boom
    assert eq.tone_match(thin)[0] == []
    dull = tgt.copy()
    dull[eq.TONE_CENTERS.index(4000)] -= 6.0                 # dull presence gets lifted...
    assert eq.tone_match(dull)[1]["tone_match_db"][4000] > 2
    snr = np.full(len(tgt), 60.0)
    snr[eq.TONE_CENTERS.index(4000)] = 20.0                  # ...unless that band is mostly hiss
    assert eq.tone_match(dull, snr_db=snr)[0] == []


def test_resonances():
    for seed, f0 in ((5, (90, 150)), (6, (100, 115)), (7, (180, 240))):   # normal, monotone, female: no room => no notches
        dry = speechlike(120, seed=seed, f0_range=f0)
        dry = dry + noise(len(dry), -80)
        assert eq.find_resonances(dry, SR, activity.analyze_activity(dry, SR))[0] == [], f0
    booth = peaking(peaking(dry, 230, 8, 6), 400, 6, 8)
    found = [h for _, h, _, _ in eq.find_resonances(booth, SR, activity.analyze_activity(booth, SR))[0]]
    assert len(found) == 2 and abs(found[0] - 230) < 12 and abs(found[1] - 400) < 15, found


def test_dereverb():
    dry = speechlike(60)
    assert dereverb.estimate_t60(dry + noise(len(dry), -80), SR) is None
    wet = reverb(dry, 0.4) + noise(len(dry), -80)
    t60 = dereverb.estimate_t60(wet, SR)
    assert t60 and abs(t60 - 0.4) < 0.08, t60
    y = dereverb.dereverb(wet, SR, t60, "medium")
    act = np.abs(dry) > 1e-4
    near = signal.convolve(act.astype(float), np.ones(int(0.06 * SR)), mode="same") > 0
    tail = (signal.convolve(act.astype(float), np.ones(int(0.6 * SR)), mode="same") > 0) & ~near
    e = lambda s, m: 10 * np.log10(np.mean(s[m] ** 2))
    assert e(y, act) - e(dry, act) > -1.0           # direct voice kept (only its reverb removed)
    assert e(y, tail) - e(wet, tail) < -3.0          # reverb tails reduced


def test_declick_strong():
    x = speechlike(10)
    x = x + noise(len(x), -70)
    act = activity.analyze_activity(x, SR)
    quiet = np.flatnonzero(~act.speech_mask)
    gap = int(quiet[len(quiet) // 2] * act.hop)
    y = x.copy()
    y[gap : gap + 90] += 0.008 * np.hanning(90) * np.sign(np.random.default_rng(0).standard_normal(90))   # faint lip smack in a pause
    assert any(s - 50 <= gap <= e for s, e in declick.find_clicks(y, SR, "strong"))


def test_fit_head_tail():
    x = np.concatenate([np.zeros(int(2.5 * SR), np.float32), speechlike(5), np.zeros(int(0.2 * SR), np.float32)])
    x = x + noise(len(x), -70)
    bounds = acx.speech_bounds(x, SR)
    y, st = roomtone.fit_head_tail(x, SR, activity.analyze_activity(x, SR), bounds, 0.75, 2.0)
    m = acx.measure(y, SR)
    assert abs(m["head_s"] - 0.75) < 0.03 and abs(m["tail_s"] - 2.0) < 0.03, m   # trimmed head, padded tail


def test_chain_passes_acx_and_mp3():
    raw = 0.2 * reverb(peaking(speechlike(30), 110, 12, 2), 0.3)          # boomy, roomy, quiet
    raw = np.concatenate([noise(SR // 4, -52), raw + noise(len(raw), -52), noise(SR // 4, -52)])
    s = PipelineSettings(mic="pd70", enable_neural=False, dereverb_strength="light")
    y, used = process_array(raw, SR, s, analyze(raw, SR))
    items = acx.check(acx.measure(y, SR))
    assert acx.passed(items), acx.summary(items)
    assert used["tone_match_db"].get(100, 0) < -2, used.get("tone_match_db")   # the 110 Hz boom got cut
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mp3")
        io_utils.save_mp3(p, y, SR)
        dec, dsr = io_utils.load_audio(p)
        items = acx.check(acx.measure(dec, dsr), mp3=True)
        assert acx.passed(items), acx.summary(items)
        kbps = os.path.getsize(p) * 8 / (len(dec) / dsr) / 1000
        assert 185 < kbps < 200, kbps   # CBR 192


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
