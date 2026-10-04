"""Smart Auto: listen to the raw file (measure it), decide what each stage needs,
and say why. Plus the microphone profiles that set each decision's starting point.

Every decision is a measured deviation from a finished-narration reference
(eq.TONE_TARGET_REL_1K for tone; ACX numbers for noise) mapped onto the
existing strength presets, so the per-stage code stays the single owner of
what "medium" or "strong" actually does.

Mic profiles describe the mic TYPE, not a lab measurement of the exact unit:
the per-file tone match measures the real recording anyway, so a profile only
needs to know what that kind of mic tends to get wrong.
  * Dynamic broadcast mic (Maono PD70): worked close, so proximity-effect
    boom and plosive thump; low output, so the Vocaster's preamp runs hot and
    adds hiss; naturally dark top end (benefits from a little presence/air,
    must not be dulled further); rejects most of the booth.
  * Large-diaphragm "vintage" condenser (Stellar X2): hears the booth -- small
    room reflections and modes -- far more; more sibilant; lower self-noise
    than a hot dynamic chain. (Not brighter in practice: see the profile.)
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .dsp import activity as act_mod, dereverb as dereverb_mod, eq as eq_mod

LEVELS = ["light", "medium", "strong", "max"]

MIC_PROFILES = {
    "generic": {
        "label": "Other / generic mic",
        "highpass_hz": 80.0, "lowend": (0, 3), "soften": (0, 2), "deess": (0, 2),
        "dereverb_bias": 0, "tone_boost_scale": 1.0, "dehiss_bias": 0,
    },
    "pd70": {
        "label": "Maono PD70 (dynamic)",
        "highpass_hz": 90.0,           # proximity effect puts real rumble/thump down there
        "lowend": (1, 3),              # at least medium bass control: boom swells with distance changes
        "soften": (0, 1),              # never "strong": a dynamic is dark already, don't dull it
        "deess": (0, 2),
        "dereverb_bias": -1,           # rejects most of the room
        "tone_boost_scale": 1.0,       # allow presence/air lift for a crisp read
        "dehiss_bias": 1,              # hot preamp gain => hiss
    },
    "stellar_x2": {
        "label": "Stellar X2 (vintage condenser)",
        # highpass / soften / tone_boost below were calibrated BY EAR on a real X2 read in the user's booth
        # (the approved "CLEAR2" master): the "bright capsule" guesses (75 Hz, soften >= medium, half
        # presence/air lift) measured ~1 dB duller at 2.5-4 kHz and ~2 dB at 10-12.5 kHz than the approved sound.
        "highpass_hz": 90.0,
        "lowend": (0, 3),
        "soften": (0, 2),              # measured per file; this X2 reads on target in presence, not bright
        "deess": (1, 2),               # at least medium de-essing
        "dereverb_bias": 1,            # hears the booth: reflections, modes
        "tone_boost_scale": 1.0,
        "dehiss_bias": 0,
    },
}


def _band(rel: dict[int, float], lo: int, hi: int) -> float:
    v = [d for c, d in rel.items() if lo <= c <= hi and np.isfinite(d)]
    return float(np.mean(v)) if v else 0.0


def _level(i: int, bounds: tuple[int, int]) -> str:
    return LEVELS[int(np.clip(i, bounds[0], bounds[1]))]


def measure_tone(mono: np.ndarray, sr: int, act: act_mod.Activity) -> dict[str, float]:
    """Excess (dB, + = too much) of each region vs the narration target, relative to 1 kHz."""
    db = eq_mod.measure_speech_bands(mono, sr, act, eq_mod.TONE_CENTERS, octave_frac=3.0)
    c = np.array(eq_mod.TONE_CENTERS)
    ref = np.nanmean(db[(c >= 800) & (c <= 1250)])
    tgt = eq_mod.TONE_TARGET_REL_1K
    rel = {cc: float(d - ref - tgt[cc]) for cc, d in zip(eq_mod.TONE_CENTERS, db)}
    return {
        "boom": _band(rel, 80, 160), "mud": _band(rel, 200, 500), "box": _band(rel, 500, 800),
        "harsh": _band(rel, 2500, 4000), "sibilance": _band(rel, 5000, 8000), "air": _band(rel, 10000, 12500),
    }


def apply_mic(settings):
    """The parts of a mic profile that apply even with Smart Auto off (manual strengths stay yours)."""
    prof = MIC_PROFILES.get(settings.mic, MIC_PROFILES["generic"])
    return replace(settings, highpass_hz=prof["highpass_hz"], tone_boost_scale=prof["tone_boost_scale"])


def decide(settings, mono: np.ndarray, sr: int, pre) -> tuple[object, list[str]]:
    """Returns (settings with every automatic choice filled in, human-readable diagnosis)."""
    prof = MIC_PROFILES.get(settings.mic, MIC_PROFILES["generic"])
    s = apply_mic(settings)
    notes = [f"Mic profile: {prof['label']}"]
    act = act_mod.analyze_activity(mono, sr)
    t = measure_tone(mono, sr, act)

    # Low end: with room fix on, the final tone match removes the STATIC excess (measured, per band);
    # the low-end stage then only cuts narrow mud peaks and controls boom SWELLS (proximity changes,
    # plosive thump), so its strength follows how boomy the voice is but stops at "strong".
    e = max(t["boom"], t["mud"])
    cap = (prof["lowend"][0], min(prof["lowend"][1], 2 if s.enable_roomfix else 3))
    s.lowend_strength = _level(0 if e < 2 else 1 if e < 6 else 2 if e < 10 else 3, cap)
    notes.append(f"Low end: boom {t['boom']:+.1f} dB, mud {t['mud']:+.1f} dB vs target -> "
                 f"{'tone match cuts the excess, ' if s.enable_roomfix else ''}bass/mud control {s.lowend_strength}")
    if t["box"] > 2:
        notes.append(f"Boxiness (500-800 Hz) {t['box']:+.1f} dB -> cut by tone match")

    s.soften_strength = _level(0 if t["harsh"] < 1.5 else 1 if t["harsh"] < 4 else 2, prof["soften"])
    s.deess_strength = _level(0 if t["sibilance"] < 0 else 1 if t["sibilance"] < 3 else 2, prof["deess"])
    notes.append(f"Presence {t['harsh']:+.1f} dB -> harshness control {s.soften_strength}; "
                 f"sibilance {t['sibilance']:+.1f} dB -> de-esser {s.deess_strength}")
    if t["air"] < -3:
        notes.append(f"Top end dull (air {t['air']:+.1f} dB) -> tone match lifts presence/air"
                     + (" (limited for this mic)" if prof["tone_boost_scale"] < 1 else ""))

    floor = pre.noise_floor_dbfs
    i = (2 if floor > -50 else 1 if floor > -62 else 0) + prof["dehiss_bias"]
    s.dehiss_strength = _level(i, (0, 2))
    notes.append(f"Noise floor {floor:.0f} dBFS -> spectral denoise auto, hiss control {s.dehiss_strength}"
                 + (", noise sample: your selection" if s.noise_region_s else ", noise sample: auto-detected room tone"))

    if pre.click_rate_per_min > 6:
        s.declick_strength = "strong"
    notes.append(f"Mouth clicks: {pre.click_rate_per_min:.0f}/min -> de-click {s.declick_strength}")

    t60 = dereverb_mod.estimate_t60(mono, sr)
    s.room_t60_s = t60
    if s.dereverb_strength == "auto":   # anything else is the user's explicit choice
        base = dereverb_mod.auto_strength(t60)
        idx = ["light", "medium", "strong"].index(base) + prof["dereverb_bias"] if base else -1
        s.dereverb_strength = "off" if idx < 0 or not s.enable_roomfix else ["light", "medium", "strong"][min(idx, 2)]
    notes.append((f"Room decay T60 {t60:.2f} s" if t60 else "Room decay: too short to measure (dry booth)")
                 + f" -> de-reverb {s.dereverb_strength if s.enable_roomfix else 'off (room fix off)'}")
    return s, notes
